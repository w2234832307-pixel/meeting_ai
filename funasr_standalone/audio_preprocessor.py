"""
音频预处理服务
用于提升ASR识别准确率
"""
import os
import subprocess
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """音频预处理器"""
    
    def __init__(self):
        """初始化预处理器并检查ffmpeg"""
        self.ffmpeg_available = self._check_ffmpeg()
        if self.ffmpeg_available:
            logger.info("✅ ffmpeg 可用，音频预处理已启用")
        else:
            logger.warning("⚠️ ffmpeg 未安装，音频预处理功能将被禁用")
    
    def _check_ffmpeg(self) -> bool:
        """检查ffmpeg是否可用"""
        try:
            subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                check=True,
                timeout=5
            )
            return True
        except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            return False
    
    def preprocess(self, input_path: str, output_path: str = None) -> str:
        """
        对音频进行预处理
        
        Args:
            input_path: 输入音频路径
            output_path: 输出音频路径（可选）
        
        Returns:
            处理后的音频路径（如果ffmpeg不可用或处理失败，返回原路径）
        """
        if not self.ffmpeg_available:
            logger.debug("ffmpeg不可用，跳过音频预处理")
            return input_path
        
        if output_path is None:
            input_file = Path(input_path)
            output_path = str(input_file.parent / f"{input_file.stem}_processed.wav")
        
        try:
            # 使用ffmpeg进行预处理
            # 1. 转换为16kHz单声道
            # 2. 降噪
            # 3. 音量归一化
            cmd = [
                "ffmpeg",
                "-i", input_path,
                "-ac", "1",              # 单声道
                "-ar", "16000",          # 16kHz采样率
                "-af", "highpass=f=200,lowpass=f=3000,afftdn=nf=-25",  # 降噪
                "-y",                    # 覆盖输出
                "-loglevel", "error",    # 只输出错误信息
                output_path
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, timeout=60)
            logger.info(f"✅ 音频预处理完成: {output_path}")
            return output_path
            
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ 音频预处理失败: {e.stderr.decode() if e.stderr else str(e)}")
            return input_path  # 降级：返回原始文件
        except subprocess.TimeoutExpired:
            logger.error(f"❌ 音频预处理超时")
            return input_path
        except Exception as e:
            logger.error(f"❌ 音频预处理异常: {e}")
            return input_path


# 全局实例
audio_preprocessor = AudioPreprocessor()
