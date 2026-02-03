"""
Pyannote 说话人分离 HTTP 客户端

用于在主服务（meeting_ai 环境）中，通过 HTTP 调用独立的
Pyannote 服务（运行在 meeting_ai_pyannote 环境）。
"""
from typing import List, Dict, Any

import requests

from app.core.config import settings
from app.core.logger import logger


class PyannoteService:
    """调用独立 Pyannote 服务的客户端"""

    def __init__(self) -> None:
        self.base_url = getattr(settings, "PYANNOTE_SERVICE_URL", "").rstrip("/")
        if self.base_url:
            logger.info(f"🌐 Pyannote 服务已配置: {self.base_url}")
        else:
            logger.info("ℹ️ 未配置 PYANNOTE_SERVICE_URL，将跳过 Pyannote 说话人分离")

    def is_available(self) -> bool:
        return bool(self.base_url)

    def diarize(self, audio_path: str, transcript: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        调用 Pyannote 服务，为 transcript 添加/覆盖 speaker_id。

        Args:
            audio_path: 服务器本地音频路径（与 FunASR 使用同一份路径）
            transcript: ASR 返回的逐字稿列表（至少包含 text/start_time/end_time）

        Returns:
            更新后的 transcript（尽量保留原字段，只是补充 speaker_id）
        """
        if not self.is_available():
            return transcript

        if not transcript:
            return transcript

        try:
            url = f"{self.base_url}/diarize"
            payload = {
                "audio_path": audio_path,
                "transcript": [
                    {
                        "text": item.get("text", ""),
                        "start_time": float(item.get("start_time", 0.0)),
                        "end_time": float(item.get("end_time", 0.0)),
                        # speaker_id 如果已有，也带过去，但 Pyannote 会覆盖
                        "speaker_id": item.get("speaker_id"),
                    }
                    for item in transcript
                ],
            }

            logger.info(f"🎤 调用 Pyannote 服务进行说话人分离: {audio_path}")
            resp = requests.post(url, json=payload, timeout=600)
            if resp.status_code != 200:
                logger.warning(f"⚠️ Pyannote 服务返回错误: {resp.status_code} - {resp.text}")
                return transcript

            data = resp.json()
            new_items = data.get("transcript", [])

            if not new_items or len(new_items) != len(transcript):
                logger.warning("⚠️ Pyannote 返回的条目数量与原 transcript 不一致，保持原结果")
                return transcript

            # 将 Pyannote 的 speaker_id 写回原 transcript（保留其他字段）
            for orig, new in zip(transcript, new_items):
                orig["speaker_id"] = new.get("speaker_id", orig.get("speaker_id"))

            logger.info("✅ Pyannote 说话人分离完成，已更新 speaker_id")
            return transcript

        except Exception as e:
            logger.error(f"❌ 调用 Pyannote 服务失败: {e}")
            return transcript


_pyannote_service_instance: PyannoteService | None = None


def get_pyannote_service() -> PyannoteService:
    """获取 Pyannote 服务单例"""
    global _pyannote_service_instance
    if _pyannote_service_instance is None:
        _pyannote_service_instance = PyannoteService()
    return _pyannote_service_instance

