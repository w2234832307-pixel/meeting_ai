#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FunASR 独立服务 - 生产级配置
端口: 8002
功能: CPU量化加速 + 自动日志记录
"""
# =============================================
# 0. 修复 datasets 兼容性问题（必须在导入其他模块之前）
# =============================================
def _fix_datasets_compatibility():
    """修复 datasets 与 modelscope 的兼容性问题"""
    try:
        import datasets
        
        # 修复 LargeList 导入
        if not hasattr(datasets, 'LargeList'):
            try:
                from datasets import LargeList
            except ImportError:
                try:
                    import pyarrow as pa
                    if hasattr(pa, 'large_list'):
                        datasets.LargeList = pa.large_list
                    elif hasattr(pa, 'LargeList'):
                        datasets.LargeList = pa.LargeList
                except Exception:
                    pass
        
        # 修复 _FEATURE_TYPES 导入（datasets 2.19+ 中可能已移除）
        try:
            from datasets.features.features import _FEATURE_TYPES
        except ImportError:
            try:
                # 尝试从新位置导入
                from datasets.features import _FEATURE_TYPES
            except ImportError:
                try:
                    # 如果不存在，创建一个兼容的占位符
                    import datasets.features.features as features_module
                    if not hasattr(features_module, '_FEATURE_TYPES'):
                        # 创建一个空的字典作为占位符
                        features_module._FEATURE_TYPES = {}
                except Exception:
                    pass
    except Exception:
        pass  # 如果 datasets 都导入不了，让后续代码自己处理错误

# 立即执行修复
_fix_datasets_compatibility()

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
from hotword_service import get_hotword_service  # ✅ 导入热词服务
from audio_preprocessor import audio_preprocessor  # ✅ 导入音频预处理
# 声纹匹配延迟加载，避免启动时的依赖错误
# from voice_matcher import get_voice_matcher

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
    
    # =================== 配置：SenseVoiceSmall 高准确率方案 ===================
    # 策略：SenseVoiceSmall 用于高准确率识别，VAD 和说话人分离独立处理
    
    # 1. SenseVoiceSmall 主模型（仅识别，不使用 spk_model）
    logger.info("📦 加载 SenseVoiceSmall 主模型（高准确率识别）...")
    asr_model = AutoModel(
        model="iic/SenseVoiceSmall",
        device=DEVICE,
        ncpu=NCPU,
        disable_update=True
    )
    logger.info("✅ SenseVoiceSmall 加载成功")
    
    # 2. VAD 模型（用于获取时间戳）
    logger.info("📦 加载 VAD 模型（时间戳分割）...")
    vad_model = AutoModel(
        model="fsmn-vad",
        device=DEVICE,
        disable_update=True
    )
    logger.info("✅ VAD 模型加载成功")
    
    # 3. 说话人识别模型（用于声纹提取和聚类）
    logger.info("📦 加载 Cam++ 说话人模型（说话人分离）...")
    speaker_model = AutoModel(
        model="iic/speech_campplus_sv_zh-cn_16k-common",
        device=DEVICE,
        disable_update=True
    )
    logger.info("✅ Cam++ 说话人模型加载成功")
    
    # =================== 旧模型配置（已注释，可回退）===================
    # # Paraformer-zh（标准模型）
    # model = AutoModel(
    #     model="paraformer-zh",
    #     vad_model="fsmn-vad",
    #     punc_model="ct-punc",
    #     spk_model="cam++",
    #     device=DEVICE,
    #     ncpu=NCPU,
    #     disable_update=True,
    #     quantize=False
    # )
    
    # # Paraformer-Large（大模型，需要12GB+显存）
    # model = AutoModel(
    #     model="iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    #     model_revision="v2.0.4",
    #     spk_model="iic/speech_campplus_sv_zh-cn_16k-common",
    #     device=DEVICE,
    #     ncpu=NCPU,
    #     disable_update=True,
    #     quantize=False
    # )
    
    logger.info("✅ 所有模型加载成功！服务就绪。")
    
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
    hotword: str = Form("")  # 外部传入的热词（可选）
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

        # === 音频预处理（可选，提升准确率3-5%）===
        if isinstance(input_data, str) and Path(input_data).exists():
            processed_input = audio_preprocessor.preprocess(input_data)
            if processed_input != input_data:
                logger.info("✅ 使用预处理后的音频")
                input_data = processed_input
        
        # === 自动加载热词 ===
        try:
            hotword_svc = get_hotword_service()
            auto_hotwords = hotword_svc.get_hotwords_string()
            
            # 合并外部传入的热词和自动加载的热词
            if hotword and auto_hotwords:
                combined_hotwords = f"{hotword} {auto_hotwords}"
            elif auto_hotwords:
                combined_hotwords = auto_hotwords
            else:
                combined_hotwords = hotword
                
            hotword_count = len(hotword_svc.get_all_hotwords())
            logger.info(f"🔥 热词已加载: {hotword_count} 个")
        except Exception as e:
            logger.warning(f"⚠️ 热词加载失败: {e}，将不使用热词")
            combined_hotwords = hotword
        
        # === 开始推理 ===
        logger.info(f"🎤 开始语音识别... (热词: {len(combined_hotwords)} 字符)")

        # ===== 步骤1：使用 VAD 获取语音段时间戳 =====
        logger.info("🎤 步骤1: VAD 语音分段...")
        vad_res = vad_model.generate(
            input=input_data,
            batch_size_s=60  # 每60秒一段
        )
        
        # 提取 VAD 分段信息
        vad_segments = []
        if vad_res and len(vad_res) > 0:
            vad_result = vad_res[0]
            vad_segments = vad_result.get("value", [])
        
        if not vad_segments or len(vad_segments) == 0:
            logger.warning("⚠️ VAD 未检测到语音段，使用全文识别")
            vad_segments = [[0, -1]]  # 使用整个音频
        
        logger.info(f"✅ VAD 检测到 {len(vad_segments)} 个语音段")
        
        # ===== 步骤2：批量提取片段并识别（优化：批量处理 + 内存缓存）=====
        logger.info("🎤 步骤2: SenseVoiceSmall 批量识别（优化版）...")
        
        # 使用临时文件路径（如果有）
        audio_file_path = str(temp_file_path) if temp_file_path else input_data
        
        # 配置：10GB显存优化
        BATCH_SIZE = 8  # 每批处理8个片段（10GB显存）
        MAX_CONCURRENT = 2  # 最多2个并发线程
        
        import subprocess
        import tempfile as tmp
        import re
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import io
        import soundfile as sf
        import numpy as np
        
        # 优化1&4: 批量提取片段到内存，避免频繁磁盘I/O
        logger.info(f"📦 批量提取 {len(vad_segments)} 个音频片段到内存...")
        segment_audio_data = {}  # {segment_idx: (audio_data, sample_rate)}
        segment_metadata = {}  # {segment_idx: (start_ms, end_ms)}
        
        def extract_segment_to_memory(idx, segment):
            """提取单个片段到内存"""
            if not isinstance(segment, list) or len(segment) < 2:
                return None, None
            
            start_ms, end_ms = segment[0], segment[1]
            
            try:
                # 使用 ffmpeg 提取到内存（通过管道）
                cmd = ["ffmpeg", "-i", audio_file_path, "-ss", str(start_ms / 1000.0)]
                
                if end_ms != -1:
                    duration = (end_ms - start_ms) / 1000.0
                    cmd.extend(["-t", str(duration)])
                
                cmd.extend([
                    "-ac", "1", "-ar", "16000",
                    "-f", "wav",
                    "-"  # 输出到stdout
                ])
                
                # 提取音频数据到内存
                result = subprocess.run(
                    cmd, 
                    check=True, 
                    capture_output=True, 
                    timeout=30
                )
                
                # 从内存中读取音频数据
                audio_io = io.BytesIO(result.stdout)
                audio_data, sample_rate = sf.read(audio_io)
                
                return (audio_data, sample_rate), (start_ms, end_ms)
                
            except Exception as e:
                logger.warning(f"⚠️ 提取片段 {idx} 失败: {e}")
                return None, None
        
        # 优化3: 并行提取片段（控制并发数）
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(extract_segment_to_memory, idx, segment): idx 
                for idx, segment in enumerate(vad_segments)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    audio_data_info, metadata = future.result()
                    if audio_data_info is not None:
                        segment_audio_data[idx] = audio_data_info
                        segment_metadata[idx] = metadata
                except Exception as e:
                    logger.warning(f"⚠️ 提取片段 {idx} 异常: {e}")
        
        logger.info(f"✅ 成功提取 {len(segment_audio_data)} 个片段到内存")
        
        # 优化2: 批量识别（分批处理，避免显存溢出）
        segment_results = []
        full_text_parts = []
        
        # 按segment_idx排序，确保顺序
        sorted_indices = sorted(segment_audio_data.keys())
        
        # 分批处理
        for batch_start in range(0, len(sorted_indices), BATCH_SIZE):
            batch_indices = sorted_indices[batch_start:batch_start + BATCH_SIZE]
            logger.info(f"🔄 批量识别片段 {batch_start+1}-{min(batch_start+BATCH_SIZE, len(sorted_indices))}/{len(sorted_indices)}")
            
            # 将内存中的音频数据写入临时文件（批量识别需要文件路径）
            batch_files = []
            batch_metadata = []
            
            for idx in batch_indices:
                audio_data, sample_rate = segment_audio_data[idx]
                start_ms, end_ms = segment_metadata[idx]
                
                # 写入临时文件（批量识别需要）
                temp_segment = tmp.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_segment.close()
                temp_segment_path = temp_segment.name
                
                sf.write(temp_segment_path, audio_data, sample_rate)
                batch_files.append(temp_segment_path)
                batch_metadata.append((idx, start_ms, end_ms))
            
            # 批量识别
            try:
                batch_results = asr_model.generate(
                    input=batch_files,
                    language="zh",
                    use_itn=True
                )
                
                # 处理批量识别结果 - 按句子切分
                for i, (idx, start_ms, end_ms) in enumerate(batch_metadata):
                    if i < len(batch_results) and batch_results[i]:
                        result_item = batch_results[i]
                        text = result_item.get("text", "").strip()
                        
                        # 清理 SenseVoice 的语言标签
                        text = re.sub(r'<\|[^|]+\|>', '', text).strip()
                        
                        # 过滤非中文内容
                        if text:
                            # 检查是否包含日文假名
                            if re.search(r'[\u3040-\u309F\u30A0-\u30FF]', text):
                                logger.debug(f"⏭️ 过滤日文内容: {text[:20]}...")
                                continue
                            # 检查是否包含韩文
                            if re.search(r'[\uAC00-\uD7AF]', text):
                                logger.debug(f"⏭️ 过滤韩文内容: {text[:20]}...")
                                continue
                            # 检查是否主要是英文单词
                            english_chars = len(re.findall(r'[a-zA-Z]', text))
                            if len(text) > 0 and english_chars / len(text) > 0.5:
                                logger.debug(f"⏭️ 过滤英文内容: {text[:20]}...")
                                continue
                        
                        if not text:
                            continue
                        
                        # 优化：按句子切分，而不是按VAD段
                        # 检查是否有timestamp信息（句子级别）
                        timestamp = result_item.get("timestamp", [])
                        sentences = result_item.get("sentences", [])
                        
                        if sentences and len(sentences) > 0:
                            # 使用句子级别的信息
                            for sent in sentences:
                                sent_text = sent.get("text", "").strip()
                                if not sent_text or len(sent_text) < 2:  # 过滤太短的句子
                                    continue
                                
                                sent_timestamp = sent.get("timestamp", [])
                                if sent_timestamp and len(sent_timestamp) >= 2:
                                    sent_start = sent_timestamp[0] / 1000.0 if isinstance(sent_timestamp[0], (int, float)) else start_ms / 1000.0
                                    sent_end = sent_timestamp[1] / 1000.0 if isinstance(sent_timestamp[1], (int, float)) else end_ms / 1000.0
                                else:
                                    # 如果没有时间戳，使用VAD段的时间，但按句子比例分配
                                    sent_start = start_ms / 1000.0
                                    sent_end = end_ms / 1000.0 if end_ms != -1 else 999999
                                
                                segment_results.append({
                                    "start_time": round(sent_start, 2),
                                    "end_time": round(sent_end, 2),
                                    "text": sent_text,
                                    "segment_idx": idx,
                                    "_audio_data": segment_audio_data[idx]  # 缓存音频数据供步骤3使用
                                })
                                full_text_parts.append(sent_text)
                        elif timestamp and len(timestamp) > 0:
                            # 使用timestamp信息按句子切分
                            # timestamp格式可能是 [[start, end, word], ...] 或 [[start, end], ...]
                            current_sentence = []
                            current_start = None
                            current_end = None
                            
                            for ts_item in timestamp:
                                if not isinstance(ts_item, list) or len(ts_item) < 2:
                                    continue
                                
                                ts_start = ts_item[0] / 1000.0 if isinstance(ts_item[0], (int, float)) else start_ms / 1000.0
                                ts_end = ts_item[1] / 1000.0 if isinstance(ts_item[1], (int, float)) else end_ms / 1000.0
                                word = ts_item[-1] if len(ts_item) > 2 else ""
                                
                                if current_start is None:
                                    current_start = ts_start
                                
                                current_sentence.append(word)
                                current_end = ts_end
                                
                                # 遇到标点符号或停顿超过0.5秒，分句
                                if word in ["。", "？", "！", ".", "?", "!"] or (len(current_sentence) > 1 and ts_start - current_end > 0.5):
                                    sent_text = "".join(current_sentence).strip()
                                    if sent_text and len(sent_text) >= 2:  # 过滤太短的句子
                                        segment_results.append({
                                            "start_time": round(current_start, 2),
                                            "end_time": round(current_end, 2),
                                            "text": sent_text,
                                            "segment_idx": idx,
                                            "_audio_data": segment_audio_data[idx]
                                        })
                                        full_text_parts.append(sent_text)
                                    current_sentence = []
                                    current_start = None
                            
                            # 处理最后一句
                            if current_sentence:
                                sent_text = "".join(current_sentence).strip()
                                if sent_text and len(sent_text) >= 2:
                                    segment_results.append({
                                        "start_time": round(current_start, 2),
                                        "end_time": round(current_end, 2),
                                        "text": sent_text,
                                        "segment_idx": idx,
                                        "_audio_data": segment_audio_data[idx]
                                    })
                                    full_text_parts.append(sent_text)
                        else:
                            # 没有句子级别信息，按标点符号切分文本
                            # 过滤太短的文本（少于3个字）
                            if len(text) < 3:
                                continue
                            
                            # 按标点符号切分
                            sentences = re.split(r'([。！？\n])', text)
                            current_sent = ""
                            sent_start = start_ms / 1000.0
                            segment_duration = (end_ms - start_ms) / 1000.0 if end_ms != -1 else 1.0
                            char_duration = segment_duration / max(len(text), 1)
                            
                            for part in sentences:
                                if not part.strip():
                                    continue
                                
                                if part in ["。", "！", "？", "\n"]:
                                    if current_sent.strip() and len(current_sent.strip()) >= 2:
                                        sent_end = sent_start + len(current_sent) * char_duration
                                        segment_results.append({
                                            "start_time": round(sent_start, 2),
                                            "end_time": round(sent_end, 2),
                                            "text": current_sent.strip(),
                                            "segment_idx": idx,
                                            "_audio_data": segment_audio_data[idx]
                                        })
                                        full_text_parts.append(current_sent.strip())
                                    sent_start = sent_end
                                    current_sent = ""
                                else:
                                    current_sent += part
                            
                            # 处理最后一句
                            if current_sent.strip() and len(current_sent.strip()) >= 2:
                                sent_end = sent_start + len(current_sent) * char_duration
                                segment_results.append({
                                    "start_time": round(sent_start, 2),
                                    "end_time": round(sent_end, 2),
                                    "text": current_sent.strip(),
                                    "segment_idx": idx,
                                    "_audio_data": segment_audio_data[idx]
                                })
                                full_text_parts.append(current_sent.strip())
                
            except Exception as e:
                logger.warning(f"⚠️ 批量识别失败: {e}，降级为单段识别")
                # 降级：单段识别
                for idx, (start_ms, end_ms) in batch_metadata:
                    audio_data, sample_rate = segment_audio_data[idx]
                    temp_segment = tmp.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_segment.close()
                    temp_segment_path = temp_segment.name
                    sf.write(temp_segment_path, audio_data, sample_rate)
                    
                    try:
                        seg_res = asr_model.generate(
                            input=temp_segment_path,
                            language="zh",
                            use_itn=True
                        )
                        
                        if seg_res and len(seg_res) > 0:
                            text = seg_res[0].get("text", "").strip()
                            text = re.sub(r'<\|[^|]+\|>', '', text).strip()
                            
                        # 降级处理：按标点符号切分
                        if len(text) < 3:
                            continue
                        
                        # 按标点符号切分
                        sentences = re.split(r'([。！？\n])', text)
                        current_sent = ""
                        sent_start = start_ms / 1000.0
                        segment_duration = (end_ms - start_ms) / 1000.0 if end_ms != -1 else 1.0
                        char_duration = segment_duration / max(len(text), 1)
                        
                        for part in sentences:
                            if not part.strip():
                                continue
                            
                            if part in ["。", "！", "？", "\n"]:
                                if current_sent.strip() and len(current_sent.strip()) >= 2:
                                    sent_end = sent_start + len(current_sent) * char_duration
                                    segment_results.append({
                                        "start_time": round(sent_start, 2),
                                        "end_time": round(sent_end, 2),
                                        "text": current_sent.strip(),
                                        "segment_idx": idx,
                                        "_audio_data": segment_audio_data[idx]
                                    })
                                    full_text_parts.append(current_sent.strip())
                                sent_start = sent_end
                                current_sent = ""
                            else:
                                current_sent += part
                        
                        # 处理最后一句
                        if current_sent.strip() and len(current_sent.strip()) >= 2:
                            sent_end = sent_start + len(current_sent) * char_duration
                            segment_results.append({
                                "start_time": round(sent_start, 2),
                                "end_time": round(sent_end, 2),
                                "text": current_sent.strip(),
                                "segment_idx": idx,
                                "_audio_data": segment_audio_data[idx]
                            })
                            full_text_parts.append(current_sent.strip())
                    except Exception as e2:
                        logger.warning(f"⚠️ 识别片段 {idx} 失败: {e2}")
                    finally:
                        try:
                            os.remove(temp_segment_path)
                        except:
                            pass
            
            finally:
                # 清理批量临时文件
                for temp_file in batch_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        
        full_text = "".join(full_text_parts)
        logger.info(f"✅ ASR 识别完成，共 {len(segment_results)} 个片段，文本长度: {len(full_text)} 字")
        
        # ===== 步骤3：说话人分离（优化：复用步骤2的音频数据）=====
        logger.info("🎤 步骤3: 说话人分离（优化：复用缓存音频）...")
        
        # 优化2: 复用步骤2提取的音频数据，避免重复提取
        from speaker_diarization import perform_speaker_diarization_with_cached_audio
        
        # 构建缓存的音频数据映射
        cached_audio_map = {
            result['segment_idx']: result.get('_audio_data')
            for result in segment_results
            if '_audio_data' in result
        }
        
        # 调用优化后的说话人分离函数（使用缓存的音频数据）
        speaker_info = perform_speaker_diarization_with_cached_audio(
            vad_segments=vad_segments,
            cached_audio_map=cached_audio_map,
            speaker_model=speaker_model,
            device=DEVICE,
            audio_file_path=audio_file_path  # 降级时使用原始文件
        )
        
        # 将说话人信息合并到识别结果
        # speaker_info 中的 speaker_id 已经是重新映射后的连续编号（0, 1, 2, 3...）
        speaker_dict = {s['segment_idx']: s['speaker_id'] for s in speaker_info if 'segment_idx' in s}
        
        # 统计哪些 segment_idx 有声纹信息
        valid_segment_indices = set(speaker_dict.keys())
        logger.debug(f"🔍 有效声纹片段索引: {sorted(valid_segment_indices)}")
        
        # 为所有片段分配说话人ID（如果某个片段没有声纹，使用最近的有声纹片段的说话人）
        for idx, result in enumerate(segment_results):
            seg_idx = result.get('segment_idx', -1)
            
            if seg_idx in speaker_dict:
                # 有声纹信息，直接使用（已经是连续编号 0, 1, 2, 3...）
                result['speaker_id'] = speaker_dict[seg_idx]
            else:
                # 没有声纹信息，找到最近的有声纹片段
                found_speaker = None
                min_distance = float('inf')
                
                # 查找最近的有效片段
                for valid_idx in valid_segment_indices:
                    distance = abs(valid_idx - seg_idx)
                    if distance < min_distance:
                        min_distance = distance
                        found_speaker = speaker_dict[valid_idx]
                
                # 如果找到了，使用该说话人ID；否则使用默认值"0"
                result['speaker_id'] = found_speaker if found_speaker is not None else "0"
        
        # 强制重新映射说话人ID，确保从0开始连续编号
        # 注意：这只是编号规范化，不影响识别结果！
        # 哪些片段属于哪个说话人是由声纹聚类算法决定的，不是写死的
        
        all_speaker_ids = set(r['speaker_id'] for r in segment_results)
        
        # 找出每个说话人ID第一次出现的时间
        first_occurrence = {}
        for result in segment_results:
            speaker_id = result['speaker_id']
            start_time = result.get('start_time', 0)
            if speaker_id not in first_occurrence or start_time < first_occurrence[speaker_id]:
                first_occurrence[speaker_id] = start_time
        
        # 按第一次出现的时间排序（第一个出现的说话人 -> 0，第二个 -> 1...）
        unique_speakers = sorted(all_speaker_ids, key=lambda x: first_occurrence[x])
        n_speakers = len(unique_speakers)
        
        # 重新映射：第一个出现的说话人 -> 0，第二个 -> 1...
        # 这只是编号规范化，不影响哪些片段属于哪个说话人
        
        speaker_remap = {old_id: str(new_id) for new_id, old_id in enumerate(unique_speakers)}
        logger.debug(f"🔍 映射关系: {speaker_remap}")
        
        for result in segment_results:
            old_id = result['speaker_id']
            result['speaker_id'] = speaker_remap[old_id]
        
        # 验证映射结果
        final_ids = sorted(set(int(r['speaker_id']) for r in segment_results))
        if final_ids != list(range(n_speakers)):
            logger.error(f"❌ 映射后ID仍不连续: {final_ids}")
        else:
            # 检查第一个片段的ID
            first_speaker_id = segment_results[0]['speaker_id'] if segment_results else "N/A"
            logger.info(f"✅ 编号规范化完成: 0-{n_speakers-1}，第一个片段 speaker_id={first_speaker_id}")
        
        logger.info(f"✅ 说话人分离完成，识别出 {n_speakers} 个说话人（基于真实声纹聚类）")
        
        # ===== 步骤4：构建最终结果 =====
        html_text = full_text  # 保持兼容性
        transcript = segment_results
        
        # 清理 transcript 中的临时字段
        for item in transcript:
            if 'segment_idx' in item:
                del item['segment_idx']
            if '_audio_data' in item:
                del item['_audio_data']  # 清理缓存的音频数据
        
        logger.info(f"✅ 最终结果: {len(transcript)} 个片段, {len(set(t['speaker_id'] for t in transcript))} 个说话人")
        
        # 兼容旧的解析逻辑（以下代码不会执行，但保留以防万一）
        if False:  # 禁用旧逻辑
            result = {}
            sentence_info = None
            
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
                    
                    # 说话人ID（SenseVoiceSmall 使用 speaker_id 字段）
                    speaker_id = str(sent.get("speaker_id", sent.get("spk", "0")))
                    
                    # ✅ 提取置信度（如果有）
                    confidence = sent.get("confidence", None)
                    
                    item = {
                        "text": text,
                        "start_time": round(start_ms / 1000.0, 2),
                        "end_time": round(end_ms / 1000.0, 2),
                        "speaker_id": speaker_id
                    }
                    
                    # 如果有置信度信息，添加到结果中
                    if confidence is not None:
                        item["confidence"] = round(confidence, 3)
                    
                    transcript.append(item)
            
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
        
        # ===== 热词后处理替换（SenseVoiceSmall 专用）=====
        # SenseVoiceSmall 不支持原生热词，需要在结果中进行文本替换
        try:
            if combined_hotwords:
                hotword_svc = get_hotword_service()
                # 读取 hotwords.json 文件获取 mappings
                import json
                hotwords_path = Path(__file__).parent / "hotwords.json"
                if hotwords_path.exists():
                    with open(hotwords_path, 'r', encoding='utf-8') as f:
                        hotwords_data = json.load(f)
                    mappings = hotwords_data.get("mappings", {})
                else:
                    mappings = {}
                
                if mappings:
                    # 合并所有映射表
                    all_mappings = {}
                    for category, mapping_dict in mappings.items():
                        if isinstance(mapping_dict, dict):
                            all_mappings.update(mapping_dict)
                    
                    if all_mappings:
                        logger.info(f"🔄 应用热词映射: {len(all_mappings)} 个")
                        
                        # 对 transcript 中的每个文本进行替换
                        for item in transcript:
                            text = item.get("text", "")
                            for oral_form, standard_form in all_mappings.items():
                                text = text.replace(oral_form, standard_form)
                            item["text"] = text
                        
                        # 同时更新 full_text
                        for oral_form, standard_form in all_mappings.items():
                            full_text = full_text.replace(oral_form, standard_form)
                        
                        logger.info("✅ 热词替换完成")
        except Exception as e:
            logger.warning(f"⚠️ 热词替换失败: {e}")
        
        # ===== 声纹识别（可选，如果声纹库为空则跳过）=====
        matched_info = {}
        try:
            # 延迟导入，避免启动时的依赖错误
            # 注意：datasets 兼容性修复已在文件开头执行
            from voice_matcher import get_voice_matcher
            
            voice_matcher = get_voice_matcher()
            if voice_matcher and voice_matcher.enabled and transcript and temp_file_path:
                logger.info("🎙️ 开始声纹匹配...")
                
                # 1. 为每个说话人提取音频片段
                speaker_segments = voice_matcher.extract_speaker_segments(
                    audio_path=str(temp_file_path),
                    transcript=transcript,
                    duration=10  # 提取10秒
                )
                
                if speaker_segments:
                    logger.info(f"✅ 提取到 {len(speaker_segments)} 个说话人的音频片段")
                    
                    # 2. 匹配说话人身份
                    matched = voice_matcher.match_speakers(
                        speaker_segments=speaker_segments,
                        threshold=0.75  # 相似度阈值75%
                    )
                    
                    if matched:
                        logger.info(f"✅ 匹配成功: {len(matched)} 个说话人")
                        
                        # 3. 替换speaker_id为真实姓名
                        transcript = voice_matcher.replace_speaker_ids(transcript, matched)
                        
                        # 添加匹配信息到返回数据
                        matched_info = {
                            speaker_id: {
                                "name": name,
                                "employee_id": employee_id,
                                "similarity": similarity
                            }
                            for speaker_id, (employee_id, name, similarity) in matched.items()
                        }
                    else:
                        logger.warning("⚠️ 未匹配到任何说话人")
                        matched_info = {}
                else:
                    logger.warning("⚠️ 未能提取说话人音频片段")
                    matched_info = {}
            else:
                matched_info = {}
                if not voice_matcher:
                    logger.warning("⚠️ 声纹匹配器未初始化")
                elif not voice_matcher.enabled:
                    logger.info("ℹ️ 声纹库为空，跳过声纹匹配")
                    
        except ImportError as e:
            logger.warning(f"⚠️ 声纹匹配模块导入失败（依赖缺失），跳过声纹匹配")
            logger.warning(f"   错误详情: {e}")
            logger.warning(f"   如需使用声纹识别，请运行以下命令：")
            logger.warning(f"   pip install 'datasets>=2.14.0' 'chromadb==0.5.0'")
            logger.warning(f"   这是 datasets 版本兼容性问题，请尝试：")
            logger.warning(f"   1. 使用兼容版本: pip install 'datasets==2.17.0' 或 'datasets==2.18.0'")
            logger.warning(f"   2. 或升级 modelscope: pip install -U modelscope")
            logger.warning(f"   3. 如果 flagembedding 需要 datasets>=2.19.0，考虑创建独立环境")
            logger.warning(f"   注意：已尝试自动修补，但可能不够，建议使用兼容版本")
            matched_info = {}
        except Exception as e:
            logger.error(f"❌ 声纹匹配失败: {e}", exc_info=True)
            matched_info = {}
        
        return {
            "code": 0,
            "msg": "success",
            "text": full_text,
            "html": html_text,
            "data": {
                "text": full_text,
                "html": html_text,
                "transcript": transcript,
                "voice_matched": matched_info if matched_info else None  # 声纹匹配结果
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


# =============================================
# 热词管理API
# =============================================

@router.get("/hotwords")
async def get_hotwords():
    """获取当前热词列表"""
    try:
        hotword_svc = get_hotword_service()
        return {
            "code": 0,
            "msg": "success",
            "data": {
                "categories": hotword_svc.get_categories(),
                "hotwords": hotword_svc.hotwords_cache,
                "stats": hotword_svc.get_stats(),
                "total": len(hotword_svc.get_all_hotwords())
            }
        }
    except Exception as e:
        logger.error(f"❌ 获取热词失败: {e}")
        return {"code": 500, "msg": str(e)}


@router.post("/hotwords/reload")
async def reload_hotwords():
    """重新加载热词配置"""
    try:
        hotword_svc = get_hotword_service()
        success = hotword_svc.reload()
        
        if success:
            return {
                "code": 0,
                "msg": "热词重载成功",
                "data": {
                    "total": len(hotword_svc.get_all_hotwords()),
                    "stats": hotword_svc.get_stats()
                }
            }
        else:
            return {"code": 500, "msg": "重载失败"}
    except Exception as e:
        logger.error(f"❌ 重载热词失败: {e}")
        return {"code": 500, "msg": str(e)}


app.include_router(router)

if __name__ == "__main__":
    logger.info("🚀 启动 HTTP 服务: http://0.0.0.0:8002")
    uvicorn.run(app, host="0.0.0.0", port=8002)