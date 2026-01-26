#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR 独立服务 - 生产级配置
端口: 8002
功能: CPU量化加速 + 自动日志记录
"""
import os
import sys
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, APIRouter
import uvicorn
import tempfile
import shutil
import gc
import torch
import highlighter

# =============================================
# 1. 日志配置 (存入 ./logs 目录)
# =============================================
# 确保 logs 目录存在
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "funasr_service.log"

# 创建 logger
logger = logging.getLogger("funasr_service")
logger.setLevel(logging.INFO)

# 格式：时间 - 级别 - 消息
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# 处理器1：控制台输出 
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)

# 处理器2：文件输出 (自动切割，防止占满磁盘)
# maxBytes=10MB (每个文件最大10M), backupCount=10 (最多保留10个文件)
file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10*1024*1024, backupCount=10, encoding='utf-8')
file_handler.setFormatter(formatter)

# 避免重复添加
if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =============================================
# 2. 模型加载 (CPU 优化)
# =============================================
try:
    from funasr import AutoModel
    logger.info("📦 正在初始化服务...")
    
    if torch.cuda.is_available():
        DEVICE = "cuda"
        logger.info("✅ 检测到可用 GPU，使用 CUDA 加速")
    else:
        DEVICE = "cpu"
        logger.info("⚠️ 未检测到 GPU，使用 CPU 模式")
        
    # 增加线程数以利用服务器的 16核 CPU
    NCPU = 8 
    
    logger.info(f"⚙️ 加载模型中... (Device: {DEVICE}, Threads: {NCPU})")
    
    # 加载模型 (开启量化 quantize=True)
    model = AutoModel(
        model="paraformer-zh",
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        spk_model="cam++",  # ✅ 启用说话人识别
        device=DEVICE,
        ncpu=NCPU,
        disable_update=True,
        quantize=False  
    )
    
    logger.info("✅ FunASR 模型加载成功！服务就绪。")
    
except Exception as e:
    logger.critical(f"❌ 模型加载失败: {e}", exc_info=True)
    sys.exit(1)

# =============================================
# 3. FastAPI 服务
# =============================================
app = FastAPI(title="FunASR Service", version="1.0.0")

router = APIRouter(prefix="/api/v1")

# 健康检查接口 (解决 404 Health 错误)
@router.get("/health")
async def health_check():
    return {"status": "ok", "message": "FunASR Service is running"}

# 支持 file 上传或 audio_url 两种输入方式
@router.post("/transcribe")
async def transcribe(
    # 1. file 改为可选
    file: UploadFile = File(None), 
    # 2. url 参数
    audio_url: str = Form(None),   
    enable_vad: bool = Form(True),
    enable_punc: bool = Form(True),
    hotword: str = Form("")
):
    temp_file_path = None
    input_data = None 

    try:
        # === 逻辑判断 ===
        if file:
            logger.info(f"📥 接收到文件上传: {file.filename}")
            suffix = Path(file.filename).suffix
            # 存临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                shutil.copyfileobj(file.file, tmp)
                temp_file_path = Path(tmp.name)
            input_data = str(temp_file_path)

        elif audio_url:
            logger.info(f"🔗 接收到音频 URL: {audio_url}")
            input_data = audio_url.strip() # 去除空格
            
        else:
            raise HTTPException(status_code=400, detail="必须提供 file 或 audio_url")

        # === 开始推理 ===
        logger.info(f"Processing... VAD:{enable_vad} | Punc:{enable_punc}")

        res = model.generate(
            input=input_data, 
            batch_size_s=300, 
            hotword=hotword,
            use_vad=enable_vad,
            use_punc=enable_punc,
            sentence_timestamp=True,
        )
        
        # 3. 结果解析（包含时间戳和说话人ID）
        full_text = ""
        html_text = ""
        transcript = []
        if res and len(res) > 0:
            result = res[0]
            full_text = result.get("text", "")

            # 高亮
            if full_text:
                logger.info("🎨 正在进行文本高亮处理...")
                html_text = highlighter.process(full_text)
            
            # 调试：打印返回的数据结构键
            logger.info(f"🔍 FunASR返回的数据字段: {list(result.keys())}")
            
            # ===== 方案1: 句子级别（带说话人） =====
            sentence_info = result.get("sentence_info", None)
            
            if sentence_info and len(sentence_info) > 0:
                logger.info(f"✅ 使用句子级别解析（含说话人识别）")
                for sent in sentence_info:
                    text = sent.get("text", "").strip()
                    if not text:
                        continue
                    
                    # 时间戳（毫秒）
                    timestamps = sent.get("timestamp", [])
                    if timestamps and len(timestamps) > 0:
                        start_ms = timestamps[0][0] if isinstance(timestamps[0], list) else 0
                        end_ms = timestamps[-1][1] if isinstance(timestamps[-1], list) else 0
                    else:
                        start_ms = 0
                        end_ms = 0
                    
                    # 说话人ID
                    speaker_id = str(sent.get("spk", "unknown"))
                    
                    transcript.append({
                        "text": text,
                        "start_time": round(start_ms / 1000.0, 2),
                        "end_time": round(end_ms / 1000.0, 2),
                        "speaker_id": speaker_id
                    })
            
            # ===== 方案2: 词级别（需要合并成句子） =====
            else:
                logger.warning("⚠️ 未检测到句子级信息，使用词级别合并")
                raw_stamp = result.get("timestamp", [])
                
                if raw_stamp and len(raw_stamp) > 0:
                    # 合并策略：遇到标点或停顿超过1秒则分句
                    current_sentence = []
                    current_start = None
                    current_end = None
                    sentence_count = 0
                    
                    for item in raw_stamp:
                        if not isinstance(item, list) or len(item) < 2:
                            continue
                        
                        t_range = item[0]
                        word = str(item[-1]).strip()
                        
                        if not isinstance(t_range, list) or len(t_range) < 2:
                            continue
                        
                        start_ms = t_range[0]
                        end_ms = t_range[1]
                        
                        # 第一个词
                        if current_start is None:
                            current_start = start_ms
                        
                        current_sentence.append(word)
                        current_end = end_ms
                        
                        # 分句条件：遇到标点符号
                        if word in ["。", "？", "！", ".", "?", "!"]:
                            sentence_text = "".join(current_sentence)
                            if sentence_text and sentence_text not in ["。", "？", "！"]:
                                sentence_count += 1
                                transcript.append({
                                    "text": sentence_text,
                                    "start_time": round(current_start / 1000.0, 2),
                                    "end_time": round(current_end / 1000.0, 2),
                                    "speaker_id": str((sentence_count - 1) % 5 + 1)  # 假设最多5个人，循环分配
                                })
                            # 重置
                            current_sentence = []
                            current_start = None
                    
                    # 处理最后一句（没有标点结尾的）
                    if current_sentence:
                        sentence_text = "".join(current_sentence)
                        if sentence_text:
                            sentence_count += 1
                            transcript.append({
                                "text": sentence_text,
                                "start_time": round(current_start / 1000.0, 2),
                                "end_time": round(current_end / 1000.0, 2),
                                "speaker_id": str((sentence_count - 1) % 5 + 1)
                            })
                    
                    logger.info(f"📝 合并完成: {len(raw_stamp)}个词 -> {len(transcript)}个句子")
                else:
                    # 完全没有时间戳信息
                    logger.warning("⚠️ 无时间戳信息，返回纯文本")
                    transcript.append({
                        "text": full_text,
                        "start_time": 0.0,
                        "end_time": 0.0,
                        "speaker_id": "1"
                    })

        logger.info(f"✅ 识别成功: {file.filename} (长度: {len(full_text)}字)")
        
        return {
            "code": 0,
            "msg": "success",
            "text": full_text,
            "html": html_text,
            "data": {
                "text": full_text,
                "html": html_text,
                "transcript": transcript
            }
        }

    except Exception as e:
        logger.error(f"❌ 识别出错: {file.filename} - {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
    finally:
        # 清理临时文件和变量
        if temp_file_path and temp_file_path.exists():
            try:
                temp_file_path.unlink()
            except Exception:
                pass

        if 'input_data' in locals(): del input_data
        if 'res' in locals(): del res
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        logger.info("🧹 内存清理完成，准备迎接下一个任务")

app.include_router(router)

if __name__ == "__main__":
    logger.info("🚀 启动 HTTP 服务: http://0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)