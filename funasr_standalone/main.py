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
import numpy as np

if not hasattr(np, 'float_'):
    np.float_ = np.float64
if not hasattr(np, 'int_'):
    np.int_ = np.int64
if not hasattr(np, 'bool_'):
    np.bool_ = bool

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.append(root_dir)

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
import subprocess
import tempfile as tmp
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
import soundfile as sf

import requests  
from urllib.parse import urlparse 
import uuid
from pyannote_diarization import perform_pyannote_diarization
from app.services.voice_service import get_voice_service

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
    
    # # 3. 说话人识别模型（用于声纹提取和聚类）
    # logger.info("📦 加载 Cam++ 说话人模型（说话人分离）...")
    # speaker_model = AutoModel(
    #     model="iic/speech_campplus_sv_zh-cn_16k-common",
    #     device=DEVICE,
    #     disable_update=True
    # )
    # logger.info("✅ Cam++ 说话人模型加载成功")
    
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
# 辅助函数：安全下载
# =============================================
def download_audio_from_url(url: str, timeout: int = 300) -> str:
    """
    下载音频到本地临时文件，返回文件路径。
    如果失败抛出异常，由调用方捕获。
    """
    try:
        # 1. 解析后缀
        parsed_url = urlparse(url)
        ext = os.path.splitext(parsed_url.path)[1]
        if not ext:
            ext = ".mp3" # 默认后缀
            
        # 2. 创建临时文件占位
        # 使用 delete=False 确保文件关闭后不会立即被删，由后续逻辑手动删
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=ext)
        temp_file_path = temp_file.name
        temp_file.close() # 立即关闭句柄，让 requests去写

        logger.info(f"⬇️ 正在下载: {url} -> {temp_file_path}")
        
        # 3. 流式下载
        with requests.get(url, stream=True, timeout=timeout) as r:
            r.raise_for_status()
            with open(temp_file_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    
        logger.info(f"✅ 下载完成: {os.path.getsize(temp_file_path) / 1024 / 1024:.2f} MB")
        return temp_file_path

    except Exception as e:
        # 如果生成了临时文件但下载失败，尝试清理
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise e

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
@router.post("/transcribe/word-level")
async def transcribe_word_level(
    file: UploadFile = File(None),  # 文件上传（可选）
    audio_path: str = Form(None),  # 音频文件路径（可选）
    audio_url: str = Form(None),   # 音频URL（可选）
    hotword: str = Form("")
) -> dict:
    """
    字级别 ASR 识别接口（用于并行处理）
    
    输入方式（三选一）：
    1. file: 文件上传
    2. audio_path: 本地文件路径
    3. audio_url: 音频URL
    
    返回字级别时间戳，格式: [{"char": "你", "start": 0.5, "end": 0.6}, ...]
    """
    from word_level_asr import extract_word_level_timestamps
    
    # 初始化变量，防止 finally 报错
    temp_file_path = None # 用于文件上传
    url_downloaded_file = None # 用于URL下载
    input_data = None
    
    try:
        # === 1. 处理输入 ===
        if file:
            logger.info(f"📥 接收到文件上传: {file.filename}")
            suffix = Path(file.filename).suffix if file.filename else ".mp3"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t_file:
                shutil.copyfileobj(file.file, t_file)
                temp_file_path = Path(t_file.name)
            input_data = str(temp_file_path)
            
        elif audio_path:
            logger.info(f"📂 接收到本地文件路径: {audio_path}")
            if not os.path.exists(audio_path):
                return {"code": 1, "msg": f"文件不存在: {audio_path}", "words": []}
            input_data = audio_path.strip()
            
        elif audio_url:
            logger.info(f"🔗 接收到音频 URL: {audio_url}")
            # ✅ 修复核心：调用辅助函数下载
            try:
                url_downloaded_file = download_audio_from_url(audio_url)
                input_data = url_downloaded_file # 将处理目标指向下载好的本地文件
            except Exception as e:
                return {"code": 1, "msg": f"URL下载失败: {str(e)}", "words": []}
                
        else:
            return {"code": 1, "msg": "参数错误", "words": []}
        
        # === 音频预处理（可选，提升准确率3-5%）===
        if isinstance(input_data, str) and Path(input_data).exists():
            processed_input = audio_preprocessor.preprocess(input_data)
            if processed_input != input_data:
                logger.info("✅ 使用预处理后的音频")
                input_data = processed_input
        
        # === 使用 VAD 分段，避免长音频显存溢出 ===
        logger.info(f"🎤 开始字级别识别（VAD分段模式）: {input_data}")
        
        # 步骤1: VAD 语音分段
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
        
        # 优化：如果片段过多，先合并短片段（减少片段数量，提升处理速度）
        if len(vad_segments) > 200:
            logger.info(f"🔧 片段过多({len(vad_segments)}个)，合并短片段以提升处理速度...")
            merged_segments = []
            current_segment = None
            MIN_SEGMENT_DURATION_MS = 5000  # 最小片段时长5秒
            MAX_GAP_MS = 2000  # 最大间隔2秒
            
            for segment in vad_segments:
                if not isinstance(segment, list) or len(segment) < 2:
                    continue
                
                start_ms, end_ms = segment[0], segment[1]
                
                if end_ms == -1:
                    if current_segment:
                        merged_segments.append(current_segment)
                    merged_segments.append(segment)
                    current_segment = None
                    continue
                
                duration_ms = end_ms - start_ms
                
                if current_segment is None:
                    if duration_ms >= MIN_SEGMENT_DURATION_MS:
                        merged_segments.append(segment)
                    else:
                        current_segment = segment
                else:
                    prev_end = current_segment[1]
                    gap_ms = start_ms - prev_end
                    
                    if gap_ms <= MAX_GAP_MS:
                        current_segment[1] = end_ms
                        merged_duration = current_segment[1] - current_segment[0]
                        if merged_duration >= MIN_SEGMENT_DURATION_MS:
                            merged_segments.append(current_segment)
                            current_segment = None
                    else:
                        if current_segment[1] != -1:
                            prev_duration = current_segment[1] - current_segment[0]
                            if prev_duration >= MIN_SEGMENT_DURATION_MS:
                                merged_segments.append(current_segment)
                            elif len(merged_segments) > 0:
                                merged_segments[-1][1] = current_segment[1]
                        if duration_ms >= MIN_SEGMENT_DURATION_MS:
                            merged_segments.append(segment)
                            current_segment = None
                        else:
                            current_segment = segment
            
            if current_segment:
                merged_duration = current_segment[1] - current_segment[0] if current_segment[1] != -1 else 999999
                if merged_duration >= 1.0:
                    merged_segments.append(current_segment)
                elif len(merged_segments) > 0:
                    merged_segments[-1][1] = current_segment[1]
            
            original_count = len(vad_segments)
            vad_segments = merged_segments
            logger.info(f"✅ 合并完成: {original_count} → {len(merged_segments)} 个片段（减少 {original_count - len(merged_segments)} 个）")
        
        # 步骤2: 批量识别并提取字级别时间戳
        audio_file_path = input_data
        
        # 优化：如果输入是 URL，先下载到本地临时文件（避免 ffmpeg 从 URL 提取片段失败）
        url_downloaded_file = None
        if isinstance(audio_file_path, str) and audio_file_path.startswith(("http://", "https://")):
            logger.info(f"🔗 检测到 URL 输入，先下载到本地临时文件...")
            try:
                import requests
                response = requests.get(audio_file_path, stream=True, timeout=300)
                response.raise_for_status()
                
                # 根据 Content-Type 或 URL 后缀确定文件扩展名
                content_type = response.headers.get("Content-Type", "")
                if "audio/mpeg" in content_type or "audio/mp3" in content_type:
                    suffix = ".mp3"
                elif "audio/mp4" in content_type or "audio/m4a" in content_type:
                    suffix = ".m4a"
                elif "audio/wav" in content_type:
                    suffix = ".wav"
                else:
                    # 从 URL 推断
                    suffix = Path(audio_file_path).suffix or ".mp3"
                
                url_downloaded_file = tmp.NamedTemporaryFile(delete=False, suffix=suffix)
                for chunk in response.iter_content(chunk_size=8192):
                    url_downloaded_file.write(chunk)
                url_downloaded_file.close()
                audio_file_path = url_downloaded_file.name
                logger.info(f"✅ URL 下载完成: {audio_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ URL 下载失败: {e}，将尝试直接从 URL 提取（可能失败）")
        
        # 配置：10GB显存优化
        BATCH_SIZE = 8  # 每批处理8个片段
        MAX_CONCURRENT = 4  # 增加到4个并发线程（提升片段提取速度）
        
        # 批量提取片段到内存
        logger.info(f"📦 批量提取 {len(vad_segments)} 个音频片段到内存...")
        segment_audio_data = {}
        segment_metadata = {}
        
        def extract_segment_to_memory(idx, segment):
            """提取单个片段到内存 (SoundFile 高速版)"""
            start_ms, end_ms = segment[0], segment[1]
            
            try:
                # 使用 SoundFile 直接读取，替代 ffmpeg 子进程
                with sf.SoundFile(audio_file_path) as f:
                    sr = f.samplerate
                    # 计算帧位置
                    start_frame = int(start_ms / 1000 * sr)
                    
                    if end_ms == -1:
                        frames_to_read = -1 # 读取到末尾
                    else:
                        frames_to_read = int((end_ms - start_ms) / 1000 * sr)
                    
                    # Seek 并读取
                    f.seek(start_frame)
                    audio_data = f.read(frames_to_read)
                    
                    # 简单兼容性处理：如果是立体声转单声道
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.mean(axis=1)
                        
                    return (audio_data, sr), (start_ms, end_ms)

            except Exception as e:
                logger.warning(f"⚠️ 提取片段 {idx} 失败: {e}")
                return None, None
        
        # 并行提取片段
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
        
        # 批量识别并提取字级别时间戳
        all_words = []
        sorted_indices = sorted(segment_audio_data.keys())
        
        # 分批处理
        for batch_start in range(0, len(sorted_indices), BATCH_SIZE):
            batch_indices = sorted_indices[batch_start:batch_start + BATCH_SIZE]
            logger.info(f"🔄 批量识别片段 {batch_start+1}-{min(batch_start+BATCH_SIZE, len(sorted_indices))}/{len(sorted_indices)}")
            
            # 将内存中的音频数据写入临时文件
            batch_files = []
            batch_metadata = []
            
            for idx in batch_indices:
                audio_data, sample_rate = segment_audio_data[idx]
                start_ms, end_ms = segment_metadata[idx]
                
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
                
                # 提取字级别时间戳并调整时间偏移
                for i, (idx, start_ms, end_ms) in enumerate(batch_metadata):
                    if i < len(batch_results) and batch_results[i]:
                        result_item = batch_results[i]
                        
                        # 调试：打印ASR结果结构
                        if i == 0 and batch_start == 0:
                            logger.debug(f"🔍 ASR结果结构: {list(result_item.keys())}")
                            if "text" in result_item:
                                logger.debug(f"🔍 文本内容: {result_item['text'][:50]}...")
                            if "timestamp" in result_item:
                                logger.debug(f"🔍 timestamp字段: {type(result_item['timestamp'])}")
                            if "sentences" in result_item:
                                logger.debug(f"🔍 sentences字段: {len(result_item.get('sentences', []))} 个句子")
                        
                        words = extract_word_level_timestamps(result_item)
                        
                        if not words and i == 0 and batch_start == 0:
                            logger.warning(f"⚠️ 片段 {idx} 未提取到字级别时间戳，ASR结果: {result_item}")
                        
                        # 调整时间戳：加上片段的起始时间
                        segment_start_sec = start_ms / 1000.0
                        for word in words:
                            word["start"] = round(word["start"] + segment_start_sec, 3)
                            word["end"] = round(word["end"] + segment_start_sec, 3)
                        
                        all_words.extend(words)
                
            except Exception as e:
                logger.warning(f"⚠️ 批量识别失败: {e}，降级为单段识别")
                # 降级：单段识别
                for i, (idx, start_ms, end_ms) in enumerate(batch_metadata):
                    if i < len(batch_files):
                        try:
                            seg_res = asr_model.generate(
                                input=batch_files[i],
                                language="zh",
                                use_itn=True
                            )
                            if seg_res and len(seg_res) > 0:
                                words = extract_word_level_timestamps(seg_res[0])
                                segment_start_sec = start_ms / 1000.0
                                for word in words:
                                    word["start"] = round(word["start"] + segment_start_sec, 3)
                                    word["end"] = round(word["end"] + segment_start_sec, 3)
                                all_words.extend(words)
                        except Exception as e2:
                            logger.warning(f"⚠️ 识别片段 {idx} 失败: {e2}")
            finally:
                # 清理批量临时文件
                for temp_file in batch_files:
                    try:
                        os.remove(temp_file)
                    except:
                        pass
        
        # 按时间排序
        all_words.sort(key=lambda x: x["start"])
        
        logger.info(f"✅ 字级别识别完成: {len(all_words)} 个字")
        return {
            "code": 0,
            "msg": "success",
            "words": all_words
        }
        
    except Exception as e:
        logger.error(f"❌ 字级别识别失败: {e}")
        return {
            "code": 1,
            "msg": str(e),
            "words": []
        }
    finally:
        # 清理临时文件
        if temp_file_path and temp_file_path.exists():
            try:
                os.remove(temp_file_path)
                logger.debug(f"🧹 已清理临时文件: {temp_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ 清理临时文件失败: {e}")
        
        # 清理 URL 下载的临时文件
        if 'url_downloaded_file' in locals() and url_downloaded_file and os.path.exists(url_downloaded_file):
            try:
                os.remove(url_downloaded_file)
                logger.debug(f"🧹 已清理 URL 下载的临时文件: {url_downloaded_file}")
            except Exception as e:
                logger.warning(f"⚠️ 清理 URL 下载的临时文件失败: {e}")


@router.post("/transcribe")
async def transcribe(
    # 1. file 改为可选
    file: UploadFile = File(None), 
    # 2. url 参数
    audio_url: str = Form(None),    
    hotword: str = Form(""),  # 外部传入的热词（可选）
    enable_speaker_diarization: bool = Form(True)  # 是否启用说话人分离
):
    # --- 第1层缩进 (4个空格) ---
    temp_file_path = None # 上传的
    url_downloaded_file = None # URL下载的
    input_data = None

    # 定义内部函数：必须与外层逻辑保持同级缩进
    def extract_segment_to_memory(idx, segment):
        """提取单个片段到内存 (SoundFile 高速版)"""
        # --- 第2层缩进 (8个空格) ---
        if not isinstance(segment, list) or len(segment) < 2:
            return None, None
        
        start_ms, end_ms = segment[0], segment[1]
        
        try:
            # 使用 SoundFile 直接读取，替代 ffmpeg 子进程
            with sf.SoundFile(audio_file_path) as f:
                sr = f.samplerate
                # 计算帧位置
                start_frame = int(start_ms / 1000 * sr)
                
                if end_ms == -1:
                    frames_to_read = -1 # 读取到末尾
                else:
                    frames_to_read = int((end_ms - start_ms) / 1000 * sr)
                
                # Seek 并读取
                f.seek(start_frame)
                audio_data = f.read(frames_to_read)
                
                # 简单兼容性处理：如果是立体声转单声道
                if len(audio_data.shape) > 1:
                    audio_data = audio_data.mean(axis=1)
                    
                return (audio_data, sr), (start_ms, end_ms)

        except Exception as e:
            logger.warning(f"⚠️ 提取片段 {idx} 失败: {e}")
            return None, None

    try:
        # === 逻辑判断 ===
        if file:
            logger.info(f"📥 接收到文件上传: {file.filename}")
            suffix = Path(file.filename).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as t_file:
                shutil.copyfileobj(file.file, t_file)
                temp_file_path = Path(t_file.name)
            input_data = str(temp_file_path)

        elif audio_url:
            logger.info(f"🔗 接收到音频 URL: {audio_url}")
            # ✅ 修复核心：这里必须下载！否则 VAD 会卡死！
            url_downloaded_file = download_audio_from_url(audio_url)
            input_data = url_downloaded_file 
            
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
        
        # ===== 优化：总是合并相邻短片段，避免过度分段和丢内容 =====
        # 无论片段多少，都进行合并优化，避免丢失内容
        if len(vad_segments) > 1:  # 只要有多个片段，就进行合并优化
            logger.info(f"🔧 优化VAD分段({len(vad_segments)}个)，合并短片段避免丢内容...")
            merged_segments = []
            current_segment = None
            
            # 动态调整：根据片段数量调整合并策略
            if len(vad_segments) > 200:
                MIN_SEGMENT_DURATION_MS = 8000  # 最小片段时长8秒
                MAX_GAP_MS = 3000  # 最大间隔3秒
            elif len(vad_segments) > 100:
                MIN_SEGMENT_DURATION_MS = 6000  # 最小片段时长6秒
                MAX_GAP_MS = 2500  # 最大间隔2.5秒
            else:
                MIN_SEGMENT_DURATION_MS = 5000  # 最小片段时长5秒
                MAX_GAP_MS = 2000  # 最大间隔2秒
            
            logger.info(f"🔧 合并策略: 最小片段{MIN_SEGMENT_DURATION_MS/1000:.1f}秒, 最大间隔{MAX_GAP_MS/1000:.1f}秒")
            
            for segment in vad_segments:
                if not isinstance(segment, list) or len(segment) < 2:
                    continue
                
                start_ms, end_ms = segment[0], segment[1]
                
                if end_ms == -1:
                    # 最后一个片段，直接添加
                    if current_segment:
                        merged_segments.append(current_segment)
                        current_segment = None
                    merged_segments.append(segment)
                    continue
                
                duration_ms = end_ms - start_ms
                
                if current_segment is None:
                    # 第一个片段
                    if duration_ms >= MIN_SEGMENT_DURATION_MS:
                        merged_segments.append(segment)
                    else:
                        current_segment = segment  # 暂存，等待合并
                else:
                    # 检查是否可以合并
                    prev_end = current_segment[1]
                    gap_ms = start_ms - prev_end
                    
                    if gap_ms <= MAX_GAP_MS:
                        # 间隔小，可以合并
                        current_segment[1] = end_ms
                        merged_duration = current_segment[1] - current_segment[0]
                        
                        # 如果合并后达到最小长度，添加到结果
                        if merged_duration >= MIN_SEGMENT_DURATION_MS:
                            merged_segments.append(current_segment)
                            current_segment = None
                    else:
                        # 间隔大，不能合并
                        # 先处理之前的片段
                        if current_segment[1] != -1:
                            prev_duration = current_segment[1] - current_segment[0]
                            if prev_duration >= MIN_SEGMENT_DURATION_MS:
                                merged_segments.append(current_segment)
                            else:
                                # 太短，强制合并到最后一个片段（避免丢内容）
                                if len(merged_segments) > 0:
                                    last_segment = merged_segments[-1]
                                    if last_segment[1] != -1:
                                        last_segment[1] = current_segment[1]
                                    logger.debug(f"🔧 将短片段合并到前一个片段，避免丢内容")
                                else:
                                    # 如果没有前面的片段，强制添加（避免丢内容）
                                    merged_segments.append(current_segment)
                        
                        # 处理当前片段
                        if duration_ms >= MIN_SEGMENT_DURATION_MS:
                            merged_segments.append(segment)
                            current_segment = None
                        else:
                            current_segment = segment
            
            # 处理最后一个暂存的片段
            if current_segment:
                merged_duration = current_segment[1] - current_segment[0] if current_segment[1] != -1 else 999999
                if merged_duration >= 1.0:
                    merged_segments.append(current_segment)
                elif len(merged_segments) > 0:
                    merged_segments[-1][1] = current_segment[1]
            
            original_count = len(vad_segments)
            vad_segments = merged_segments
            logger.info(f"✅ 合并完成: {original_count} → {len(merged_segments)} 个片段")
        
        # ===== 步骤2：批量提取片段并识别（优化：批量处理 + 内存缓存）=====
        logger.info("🎤 步骤2: SenseVoiceSmall 批量识别（优化版）...")
        
        # 这里的 audio_file_path 必须确保是本地路径（前面逻辑已保证）
        audio_file_path = input_data
        
        # 配置：10GB显存优化
        BATCH_SIZE = 8  
        MAX_CONCURRENT = 4 
        
        # 批量提取片段到内存
        logger.info(f"📦 批量提取 {len(vad_segments)} 个音频片段到内存...")
        segment_audio_data = {}  # {segment_idx: (audio_data, sample_rate)}
        segment_metadata = {}  # {segment_idx: (start_ms, end_ms)}
        
        # 并行提取片段 (使用上面定义的内部函数)
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
        
        # 批量识别并提取字级别时间戳
        all_words = []
        sorted_indices = sorted(segment_audio_data.keys())
        segment_results = []
        full_text_parts = []
        
        # 分批处理
        for batch_start in range(0, len(sorted_indices), BATCH_SIZE):
            batch_indices = sorted_indices[batch_start:batch_start + BATCH_SIZE]
            logger.info(f"🔄 批量识别片段 {batch_start+1}-{min(batch_start+BATCH_SIZE, len(sorted_indices))}/{len(sorted_indices)}")
            
            # 将内存中的音频数据写入临时文件
            batch_files = []
            batch_metadata = []
            
            for idx in batch_indices:
                audio_data, sample_rate = segment_audio_data[idx]
                start_ms, end_ms = segment_metadata[idx]
                
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
                        
                        if not text or len(text) < 2:
                            continue
                            
                        # 检查是否有timestamp信息（句子级别）
                        sentences = result_item.get("sentences", [])
                        
                        if sentences and len(sentences) > 0:
                            for sent in sentences:
                                sent_text = sent.get("text", "").strip()
                                if not sent_text: continue
                                sent_timestamp = sent.get("timestamp", [])
                                if sent_timestamp and len(sent_timestamp) >= 2:
                                    sent_start = sent_timestamp[0] / 1000.0 if isinstance(sent_timestamp[0], (int, float)) else start_ms / 1000.0
                                    sent_end = sent_timestamp[1] / 1000.0 if isinstance(sent_timestamp[1], (int, float)) else end_ms / 1000.0
                                else:
                                    sent_start = start_ms / 1000.0
                                    sent_end = end_ms / 1000.0 if end_ms != -1 else 999999
                                
                                segment_results.append({
                                    "start_time": round(sent_start, 2),
                                    "end_time": round(sent_end, 2),
                                    "text": sent_text,
                                    "segment_idx": idx,
                                    "_audio_data": segment_audio_data[idx]
                                })
                                full_text_parts.append(sent_text)
                        else:
                            # 降级：按标点切分
                            sentences_split = re.split(r'([。！？\n])', text)
                            current_sent = ""
                            sent_start = start_ms / 1000.0
                            segment_duration = (end_ms - start_ms) / 1000.0 if end_ms != -1 else 1.0
                            char_duration = segment_duration / max(len(text), 1)
                            
                            for part in sentences_split:
                                if not part.strip(): continue
                                if part in ["。", "！", "？", "\n"]:
                                    if current_sent.strip():
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
                            if current_sent.strip():
                                segment_results.append({
                                    "start_time": round(sent_start, 2),
                                    "end_time": round(sent_start + len(current_sent) * char_duration, 2),
                                    "text": current_sent.strip(),
                                    "segment_idx": idx,
                                    "_audio_data": segment_audio_data[idx]
                                })
                                full_text_parts.append(current_sent.strip())

            except Exception as e:
                logger.warning(f"⚠️ 批量识别失败: {e}")
            finally:
                for temp_file in batch_files:
                    try: os.remove(temp_file)
                    except: pass
        
        full_text = "".join(full_text_parts)
        logger.info(f"✅ ASR 识别完成，共 {len(segment_results)} 个片段")
        
        # ===== 步骤3：说话人分离与熟人识别 (黄金组合) =====
        if not enable_speaker_diarization:
            logger.info("ℹ️ 说话人分离已禁用")
            for result in segment_results:
                result['speaker_id'] = '0'
        else:
            try:
                # 3.1 使用 Pyannote 进行高精度时间切分和单场聚合
                logger.info("🎤 步骤3.1: 调用 Pyannote 进行精准说话人分离...")
                
                # segment_results 里已经有了 start_time, end_time 和 text，正是 Pyannote 需要的格式
                # audio_file_path 是已经下载好并经 ffmpeg 处理过的本地 WAV 文件，速度极快
                diarized_transcript = perform_pyannote_diarization(
                    audio_path=audio_file_path,
                    transcript=segment_results
                )
                
                # 3.2 使用 Cam++ 进行跨场次声纹库对比 (认人)
                logger.info("🎤 步骤3.2: 调用 Cam++ 进行声纹库检索匹配...")
                try:
                    voice_svc = get_voice_service()
                    if voice_svc and voice_svc.enabled:
                        # 截取每个 SPEAKER_XX 的 10 秒纯净录音片段
                        speaker_audio_files = voice_svc.extract_speaker_segments(
                            audio_path=audio_file_path,
                            transcript=diarized_transcript,
                            duration=10
                        )
                        
                        # 去 ChromaDB 里搜索比对，相似度大于 0.75 判定为熟人
                        matched_results = voice_svc.match_speakers(
                            speaker_segments=speaker_audio_files, 
                            threshold=0.75
                        )
                        
                        # 如果匹配成功，将 SPEAKER_00 替换为 "张三"
                        segment_results = voice_svc.replace_speaker_ids(
                            transcript=diarized_transcript, 
                            matched=matched_results
                        )
                        logger.info("✅ 声纹熟人识别全流程完成")
                    else:
                        logger.info("ℹ️ 声纹库未配置或为空，跳过熟人检索，保留 SPEAKER_XX 标签")
                        segment_results = diarized_transcript
                except Exception as ve:
                    logger.warning(f"⚠️ Cam++ 声纹检索环节出现异常，降级保留原标签: {ve}")
                    segment_results = diarized_transcript
                    
            except Exception as e:
                logger.error(f"❌ 说话人分离全流程失败: {e}", exc_info=True)
                # 终极降级保障，保证接口不报错，会议纪要能出文字
                for result in segment_results:
                    if 'speaker_id' not in result:
                        result['speaker_id'] = '0'

        # ===== 步骤4：构建结果 =====
        transcript = segment_results
        for item in transcript:
            if 'segment_idx' in item: del item['segment_idx']
            if '_audio_data' in item: del item['_audio_data']
            
        return {
            "code": 0,
            "msg": "success",
            "text": full_text,
            "html": full_text,
            "data": {
                "text": full_text,
                "html": full_text,
                "transcript": transcript
            }
        }

    except Exception as e:
        logger.error(f"❌ 识别出错: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal Server Error")
        
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try: os.remove(temp_file_path)
            except: pass
        if url_downloaded_file and os.path.exists(url_downloaded_file):
            try: os.remove(url_downloaded_file)
            except: pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("🧹 清理完成")


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