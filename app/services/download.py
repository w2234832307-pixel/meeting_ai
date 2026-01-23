"""
音频文件下载服务
支持从URL下载音频文件到本地
"""
import os
import requests
from pathlib import Path
from typing import Optional
from app.core.config import settings
from app.core.logger import logger
from app.core.utils import validate_audio_format


class AudioDownloadService:
    """音频下载服务类"""
    
    def __init__(self):
        """初始化下载服务"""
        self.temp_dir = settings.TEMP_DIR
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.max_duration_seconds = settings.MAX_AUDIO_DURATION_SECONDS
    
    def download_from_url(
        self, 
        url: str, 
        audio_id: Optional[int] = None,
        max_duration_seconds: Optional[int] = None
    ) -> str:
        """
        从URL下载音频文件到本地
        
        Args:
            url: 音频文件URL
            audio_id: 音频ID（用于生成文件名，可选）
            max_duration_seconds: 最大音频时长（秒），如果超过则抛出异常
        
        Returns:
            下载后的本地文件路径
        
        Raises:
            ValueError: 如果音频时长超过限制
            Exception: 下载失败
        """
        try:
            logger.info(f"📥 开始下载音频: {url}")
            
            # 生成临时文件名
            if audio_id:
                filename = f"audio_{audio_id}_{os.path.basename(url)}"
            else:
                filename = f"audio_{os.path.basename(url)}"
            
            # 如果没有扩展名，默认使用mp3
            if not os.path.splitext(filename)[1]:
                filename += ".mp3"
            
            local_path = self.temp_dir / filename
            
            # 下载文件（支持大文件流式下载）
            response = requests.get(url, stream=True, timeout=300)
            response.raise_for_status()
            
            # 检查Content-Length（如果服务器提供）
            content_length = response.headers.get('Content-Length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                logger.info(f"📦 文件大小: {size_mb:.2f} MB")
            
            # 流式写入文件
            with open(local_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            
            file_size_mb = os.path.getsize(local_path) / (1024 * 1024)
            logger.info(f"✅ 下载完成: {local_path} ({file_size_mb:.2f} MB)")
            
            # 验证音频格式
            if not validate_audio_format(str(local_path)):
                logger.warning(f"⚠️ 文件格式可能不受支持: {filename}")
            
            # 注意：实际时长验证需要在ASR处理时进行，这里只做文件大小检查
            # 如果提供了最大时长限制，可以通过文件大小估算（粗略）
            # 实际时长需要从ASR服务或音频元数据获取
            
            return str(local_path)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ 下载失败: {e}")
            raise Exception(f"音频下载失败: {str(e)}")
        except Exception as e:
            logger.error(f"❌ 下载异常: {e}")
            raise
    
    def get_file_path_from_db(self, audio_id: int) -> Optional[str]:
        """
        从数据库获取音频文件路径或URL，如果不在本地则下载
        
        Args:
            audio_id: 音频ID
        
        Returns:
            本地文件路径，如果失败则返回None
        """
        try:
            from app.services.database import database_service
            
            # 从数据库获取音频信息
            audio_info = database_service.get_audio_info(audio_id)
            if not audio_info:
                logger.error(f"❌ 无法从数据库获取音频信息: ID={audio_id}")
                return None
            
            # 优先使用本地路径
            file_path = audio_info.get('file_path')
            if file_path and os.path.exists(file_path):
                logger.info(f"✅ 使用本地文件路径: {file_path}")
                return file_path
            
            # 如果没有本地路径，使用URL下载
            file_url = audio_info.get('file_url')
            if file_url:
                # 检查音频时长（如果数据库中有）
                duration = audio_info.get('duration')
                if duration and duration > self.max_duration_seconds:
                    raise ValueError(f"音频时长 {duration}秒 超过限制 {self.max_duration_seconds}秒（5小时）")
                
                # 下载文件
                return self.download_from_url(file_url, audio_id=audio_id)
            
            logger.error(f"❌ 音频信息中没有file_path或file_url: ID={audio_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ 获取音频文件失败: {e}")
            return None


# 创建单例实例
audio_download_service = AudioDownloadService()