#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立的 Pyannote 说话人分离服务

用法（在 meeting_ai_pyannote 环境中）：

    conda activate meeting_ai_pyannote
    cd /home/ubuntu/meeting_ai/funasr_standalone
    
    # 方式1：直接运行（推荐，自动读取 .env 中的 HF_TOKEN）
    python pyannote_server.py
    
    # 方式2：使用 uvicorn 命令（如果需要自定义参数）
    uvicorn pyannote_server:app --host 0.0.0.0 --port 8100

配置说明：
- HF_TOKEN: 在 funasr_standalone/.env 文件中配置，或通过环境变量
- PYANNOTE_HOST: 服务监听地址（默认 0.0.0.0）
- PYANNOTE_PORT: 服务端口（默认 8100）

FunASR 主服务再通过 HTTP 调用本服务的 /diarize 接口即可。
"""
import os
from typing import List
from pathlib import Path

from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv

from pyannote_diarization import perform_pyannote_diarization

# 加载 .env 文件（如果存在）
env_path = Path(__file__).parent / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path)


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


app = FastAPI(title="Pyannote Diarization Service", version="1.0.0")


@app.post("/diarize", response_model=DiarizeResponse)
async def diarize(req: DiarizeRequest) -> DiarizeResponse:
    """
    使用 Pyannote.audio 对给定音频和转写结果进行说话人分离。

    - 输入：服务器本地音频路径 + 基于该音频的 transcript（含起止时间）
    - 输出：在 transcript 基础上补充 / 覆盖 speaker_id 字段
    """
    # 从环境变量中读取 HF_TOKEN（优先从 .env 文件，其次从环境变量）
    # 注意：.env 文件应该在 funasr_standalone 目录下
    # 已经在文件开头加载了 .env，这里直接读取即可
    hf_token = os.getenv("HF_TOKEN") or None

    # 将 Pydantic 对象转换为普通 dict 以复用现有逻辑
    transcript_dicts: List[dict] = [
        {
            "text": item.text,
            "start_time": item.start_time,
            "end_time": item.end_time,
            # 如果已有 speaker_id，会在内部被覆盖成 Pyannote 的结果
            "speaker_id": item.speaker_id,
        }
        for item in req.transcript
    ]

    updated = perform_pyannote_diarization(
        audio_path=req.audio_path,
        transcript=transcript_dicts,
        use_auth_token=hf_token,
    )

    # 再次封装为 Pydantic 模型
    resp_items: List[TranscriptItem] = [
        TranscriptItem(
            text=item.get("text", ""),
            start_time=float(item.get("start_time", 0.0)),
            end_time=float(item.get("end_time", 0.0)),
            speaker_id=str(item.get("speaker_id")) if item.get("speaker_id") is not None else None,
        )
        for item in updated
    ]

    return DiarizeResponse(transcript=resp_items)


# ============================================
# 直接启动服务（无需手动运行 uvicorn 命令）
# ============================================
if __name__ == "__main__":
    import uvicorn
    
    # 从 .env 读取配置（已经在文件开头加载了）
    host = os.getenv("PYANNOTE_HOST", "0.0.0.0")
    port = int(os.getenv("PYANNOTE_PORT", "8100"))
    
    print(f"🚀 启动 Pyannote 说话人分离服务...")
    print(f"   地址: http://{host}:{port}")
    print(f"   文档: http://{host}:{port}/docs")
    print(f"   HF_TOKEN: {'已配置' if os.getenv('HF_TOKEN') else '未配置（可能无法加载模型）'}")
    print(f"\n按 Ctrl+C 停止服务\n")
    
    uvicorn.run(
        "pyannote_server:app",
        host=host,
        port=port,
        reload=False,  # 生产环境建议关闭自动重载
        log_level="info"
    )

