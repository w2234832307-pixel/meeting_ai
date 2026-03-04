#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的 Pyannote 说话人分离服务
"""
import os
import logging
from typing import List, Optional
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio
from fastapi import BackgroundTasks
import torch
import soundfile as sf
import subprocess
import tempfile

# 强制离线：必须在导入 pyannote/huggingface 相关模块前设置，避免任何联网 HEAD/下载
os.environ["HF_HUB_OFFLINE"] = os.getenv("HF_HUB_OFFLINE", "1") or "1"
os.environ["TRANSFORMERS_OFFLINE"] = os.getenv("TRANSFORMERS_OFFLINE", "1") or "1"

# 确保 pyannote_diarization.py 在同一目录下
from pyannote_diarization import perform_pyannote_diarization, get_pyannote_pipeline, process_audio_with_pipeline

# 配置日志格式
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("PyannoteServer")

# 加载 .env 文件
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)

# ================= 数据模型 =================

class TranscriptItem(BaseModel):
    text: str
    start_time: float
    end_time: float
    speaker_id: str | None = None

class DiarizeRequest(BaseModel):
    audio_path: str
    transcript: List[TranscriptItem]

class DiarizeResponse(BaseModel):
    transcript: List[TranscriptItem]

class RTTMRequest(BaseModel):
    audio_path: str

class RTTMResponse(BaseModel):
    rttm: str
    error: Optional[str] = None

# ================= APP 定义 =================

app = FastAPI(title="Pyannote Diarization Service", version="1.0.0")


@app.on_event("startup")
async def startup_event():
    """FastAPI 启动时预加载 pipeline（全局单例）"""
    logger.info("🚀 启动时预加载 Pyannote pipeline...")
    hf_token = os.getenv("HF_TOKEN") or None
    pipeline = get_pyannote_pipeline(use_auth_token=hf_token)
    if pipeline:
        logger.info("✅ Pyannote pipeline 预加载成功（全局单例，后续请求直接使用）")
    else:
        logger.warning("⚠️ Pyannote pipeline 预加载失败，将在首次请求时加载")


def _extract_annotation(diarization):
    """
    兼容不同版本 Pyannote：
    - 旧版: pipeline(...) 直接返回支持 itertracks 的 Annotation
    - 3.x: 可能返回 DiarizeOutput，真正的 Annotation 在某个属性里
    """
    # 旧版：直接就是 Annotation
    if hasattr(diarization, "itertracks"):
        return diarization

    # 尝试通过 __dict__ 或 vars() 访问（适用于 dataclass/NamedTuple）
    try:
        obj_dict = vars(diarization) if hasattr(diarization, "__dict__") else {}
        for key, value in obj_dict.items():
            if value is not None and hasattr(value, "itertracks"):
                return value
    except:
        pass

    # 尝试通过索引访问（如果是 NamedTuple）
    try:
        if hasattr(diarization, "__len__"):
            for i in range(len(diarization)):
                ann = diarization[i]
                if ann is not None and hasattr(ann, "itertracks"):
                    return ann
    except:
        pass

    # 新版：DiarizeOutput dataclass - 尝试多种可能的属性名
    possible_attrs = ["annotation", "speaker", "output", "result", "diarization", "labels"]
    ann = None
    
    for attr in possible_attrs:
        try:
            ann = getattr(diarization, attr, None)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
        except:
            continue
    
    # 如果是 dataclass，尝试访问所有字段
    if hasattr(diarization, "__dataclass_fields__"):
        for field_name in diarization.__dataclass_fields__.keys():
            try:
                ann = getattr(diarization, field_name, None)
                if ann is not None and hasattr(ann, "itertracks"):
                    return ann
            except:
                continue
    
    # 尝试 dir() 查找所有属性
    for attr_name in dir(diarization):
        if attr_name.startswith("_"):
            continue
        try:
            ann = getattr(diarization, attr_name)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
        except:
            continue

    # 有些版本可能返回 dict
    if isinstance(diarization, dict):
        for key in ["annotation", "speaker", "output", "result", "diarization"]:
            ann = diarization.get(key)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann

    # 如果还是找不到，打印所有属性以便调试
    attrs = [a for a in dir(diarization) if not a.startswith("_")]
    try:
        obj_dict = vars(diarization) if hasattr(diarization, "__dict__") else {}
        logger.error(f"DiarizeOutput 对象详情: {obj_dict}")
    except:
        pass
    
    raise TypeError(
        f"Unsupported diarization output type: {type(diarization)}\n"
        f"Available attributes: {attrs}\n"
        f"Type: {type(diarization).__name__}\n"
        f"请检查 Pyannote 版本和 DiarizeOutput 的实际结构"
    )

@app.post("/rttm", response_model=RTTMResponse)
async def get_rttm(
    audio_path: Optional[str] = Form(None),
    audio_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
) -> RTTMResponse:
    """
    获取 RTTM 格式结果（用于并行处理）
    支持三种输入方式：
    1. 文件上传: file
    2. URL: audio_url
    3. 本地路径: audio_path (仅当服务端可访问时)
    """
    hf_token = os.getenv("HF_TOKEN") or None

    try:
        # 获取 pipeline (优先读 offline_config.yaml)
        pipeline = get_pyannote_pipeline(use_auth_token=hf_token)

        if pipeline is None:
            return RTTMResponse(rttm="", error="Failed to load Pyannote pipeline")

        # 确定音频来源
        audio_path_to_use: Optional[str] = None
        tmp_path: Optional[str] = None
        file_id: str = "audio"

        try:
            # 优先级：文件上传 > URL > 本地路径
            if file is not None:
                import tempfile
                suffix = Path(file.filename or "audio.mp3").suffix or ".mp3"
                file_id = Path(file.filename or "audio").stem
                logger.info(f"📤 [RTTM请求] 接收文件上传: {file.filename}")
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    tmp_path = tmp.name
                audio_path_to_use = tmp_path
                logger.info(f"✅ 文件已保存到临时文件: {tmp_path}")
                
            elif audio_url:
                import requests
                import tempfile
                
                logger.info(f"🔗 [RTTM请求] 检测到音频 URL，正在服务器端下载: {audio_url}")
                file_id = Path(audio_url).stem
                resp = requests.get(audio_url, timeout=300, stream=True)
                resp.raise_for_status()

                suffix = Path(audio_url).suffix or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = tmp.name
                audio_path_to_use = tmp_path
                logger.info(f"✅ 音频已下载到临时文件: {tmp_path}")
                
            elif audio_path:
                # 检查是否是 URL（兼容旧接口）
                if audio_path.startswith(("http://", "https://")):
                    import requests
                    import tempfile
                    
                    logger.info(f"🔗 [RTTM请求] 检测到音频 URL（旧接口格式），正在服务器端下载: {audio_path}")
                    file_id = Path(audio_path).stem
                    resp = requests.get(audio_path, timeout=300, stream=True)
                    resp.raise_for_status()

                    suffix = Path(audio_path).suffix or ".mp3"
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                        for chunk in resp.iter_content(chunk_size=8192):
                            if chunk:
                                tmp.write(chunk)
                        tmp_path = tmp.name
                    audio_path_to_use = tmp_path
                    logger.info(f"✅ 音频已下载到临时文件: {tmp_path}")
                else:
                    # 本地路径（仅当服务端可访问时）
                    if os.path.exists(audio_path):
                        audio_path_to_use = audio_path
                        file_id = Path(audio_path).stem
                        logger.info(f"📂 [RTTM请求] 使用本地路径: {audio_path}")
                    else:
                        raise FileNotFoundError(
                            f"音频文件不存在（可能是跨系统路径）: {audio_path}。"
                            f"请使用文件上传（file）或 URL（audio_url）方式。"
                        )
            else:
                return RTTMResponse(rttm="", error="缺少音频输入：请提供 file、audio_url 或 audio_path 之一")

            if not audio_path_to_use or not os.path.exists(audio_path_to_use):
                raise FileNotFoundError(f"音频文件不存在: {audio_path_to_use}")

            # 开始推理：手动解码音频，绕过 pyannote 内部 AudioDecoder
            # 如果格式不支持（如 M4A），使用 ffmpeg 转换
            converted_audio_path = None
            try:
                # 尝试直接读取
                data, sample_rate = sf.read(audio_path_to_use)
            except Exception as e:
                # 如果 soundfile 不支持该格式，使用 ffmpeg 转换为 WAV
                logger.info(f"⚠️ soundfile 不支持该格式，使用 ffmpeg 转换: {audio_path_to_use}")
                try:
                    # 使用 ffmpeg 转换为 WAV
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                        converted_audio_path = tmp_wav.name
                    
                    cmd = [
                        "ffmpeg", "-i", audio_path_to_use,
                        "-ac", "1",  # 单声道
                        "-ar", "16000",  # 16kHz 采样率
                        "-f", "wav",
                        "-y",  # 覆盖输出文件
                        converted_audio_path
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=60
                    )
                    
                    # 读取转换后的 WAV 文件
                    data, sample_rate = sf.read(converted_audio_path)
                    logger.info(f"✅ 音频格式转换成功: {converted_audio_path}")
                except subprocess.CalledProcessError as ffmpeg_error:
                    error_msg = ffmpeg_error.stderr.decode() if ffmpeg_error.stderr else str(ffmpeg_error)
                    raise RuntimeError(f"ffmpeg 转换失败: {error_msg}") from e
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg 未安装，无法处理 M4A 等格式。请安装: apt-get install ffmpeg 或 conda install ffmpeg") from e
            # soundfile 返回的是 [T] 或 [T, C]，而 pyannote 期望的是 [C, T]
            if data.ndim == 1:
                data = data[None, :]
            else:
                data = data.T  # (channels, time)

            waveform = torch.tensor(data, dtype=torch.float32)
            
            # 使用公共函数处理音频（支持长音频分段处理）
            diarization = process_audio_with_pipeline(pipeline, waveform, sample_rate)

        finally:
            # 清理临时文件（如果有）
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.info(f"🧹 已清理 RTTM 临时音频文件: {tmp_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理 RTTM 临时音频文件失败: {e}")
            
            # 清理格式转换产生的临时文件
            if 'converted_audio_path' in locals() and converted_audio_path and os.path.exists(converted_audio_path):
                try:
                    os.remove(converted_audio_path)
                    logger.debug(f"🧹 已清理格式转换临时文件: {converted_audio_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理格式转换临时文件失败: {e}")

        # 兼容不同版本输出，拿到真正的 Annotation
        annotation = _extract_annotation(diarization)

        # 格式化输出
        # 为了后续处理简单统一，这里强制将所有说话人标签转换为标准格式 SPEAKER_XX
        rttm_lines = []
        label_to_index: dict = {}
        next_speaker_index = 0

        for turn, _, speaker in annotation.itertracks(yield_label=True):
            start = turn.start
            duration = turn.end - turn.start

            # 原始标签可能为 None、空字符串、整数、'A'/'B' 等各种形式
            raw_label = speaker if speaker not in (None, "") else f"UNK_{next_speaker_index}"

            # 统一映射到连续的整数 ID
            if raw_label not in label_to_index:
                label_to_index[raw_label] = next_speaker_index
                next_speaker_index += 1

            speaker_idx = label_to_index[raw_label]
            speaker_str = f"SPEAKER_{speaker_idx:02d}"

            rttm_lines.append(
                f"SPEAKER {file_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker_str} <NA> <NA>"
            )

        rttm_content = "\n".join(rttm_lines)
        logger.info(f"✅ RTTM 生成成功 ({len(rttm_lines)} 个片段)")
        return RTTMResponse(rttm=rttm_content)

    except Exception as e:
        logger.error(f"❌ RTTM 生成失败: {e}", exc_info=True)
        return RTTMResponse(rttm="", error=str(e))


gpu_semaphore = asyncio.Semaphore(2)

@app.post("/diarize", response_model=DiarizeResponse)
async def diarize(req: DiarizeRequest) -> DiarizeResponse:
    """
    标准说话人分离接口（合并 Transcript）- 异步防阻塞改造版
    """
    hf_token = os.getenv("HF_TOKEN") or None
    
    logger.info(f"📂 [Diarize请求] 音频: {req.audio_path}, 字幕条数: {len(req.transcript)}")
    logger.info("⏳ 正在等待进入处理队列...")

    # 1. 转换为字典列表
    transcript_dicts: List[dict] = [
        {
            "text": item.text,
            "start_time": item.start_time,
            "end_time": item.end_time,
            "speaker_id": item.speaker_id,
        }
        for item in req.transcript
    ]

    # 2. 申请锁并放入后台线程执行
    async with gpu_semaphore:
        logger.info("获取到处理令牌，开始执行说话人分离 (后台线程)...")
        updated = await asyncio.to_thread(
            perform_pyannote_diarization,
            req.audio_path,
            transcript_dicts,
            hf_token
        )
        logger.info("🔓 处理完成，释放锁。下一个任务可以开始了。")

    # 3. 封装返回
    resp_items: List[TranscriptItem] = [
        TranscriptItem(
            text=item.get("text", ""),
            start_time=float(item.get("start_time", 0.0)),
            end_time=float(item.get("end_time", 0.0)),
            speaker_id=str(item.get("speaker_id")) if item.get("speaker_id") else None,
        )
        for item in updated
    ]

    return DiarizeResponse(transcript=resp_items)


if __name__ == "__main__":
    import uvicorn
    
    host = os.getenv("PYANNOTE_HOST", "0.0.0.0")
    port = int(os.getenv("PYANNOTE_PORT", "8100"))
    
    print(f"\n🚀 Pyannote 服务启动中...")
    print(f"👉 监听: http://{host}:{port}")
    
    uvicorn.run(
        "pyannote_server:app",
        host=host,
        port=port,
        reload=False,
        log_level="info"
    )