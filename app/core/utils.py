"""
通用工具函数
包含重试机制、超时控制等工业级功能
"""
import time
import functools
from typing import Callable, TypeVar, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError
)
from app.core.logger import logger

T = TypeVar('T')


def retry_with_backoff(
    max_attempts: int = 3,
    initial_wait: float = 1.0,
    max_wait: float = 10.0,
    exponential_base: float = 2.0,
    retry_on: tuple = (Exception,)
):
    """
    带指数退避的重试装饰器
    
    Args:
        max_attempts: 最大重试次数
        initial_wait: 初始等待时间（秒）
        max_wait: 最大等待时间（秒）
        exponential_base: 指数基数
        retry_on: 需要重试的异常类型元组
    
    Returns:
        装饰器函数
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        @retry(
            stop=stop_after_attempt(max_attempts),
            wait=wait_exponential(
                multiplier=initial_wait,
                max=max_wait,
                exp_base=exponential_base
            ),
            retry=retry_if_exception_type(retry_on),
            reraise=True
        )
        def wrapper(*args: Any, **kwargs: Any) -> T:
            try:
                return func(*args, **kwargs)
            except RetryError as e:
                logger.error(f"函数 {func.__name__} 重试 {max_attempts} 次后仍失败: {e}")
                raise
            except Exception as e:
                logger.warning(f"函数 {func.__name__} 执行失败，准备重试: {e}")
                raise
        
        return wrapper
    return decorator


def validate_file_size(file_path: str, max_size_mb: int = 500) -> bool:
    """
    验证文件大小
    
    Args:
        file_path: 文件路径
        max_size_mb: 最大文件大小（MB）
    
    Returns:
        是否通过验证
    """
    import os
    if not os.path.exists(file_path):
        return False
    
    size_mb = os.path.getsize(file_path) / (1024 * 1024)
    return size_mb <= max_size_mb


def validate_audio_format(filename: str) -> bool:
    """
    验证音频文件格式
    
    Args:
        filename: 文件名
    
    Returns:
        是否为支持的格式
    """
    supported_formats = {'.mp3', '.wav', '.m4a', '.aac', '.flac', '.opus', '.amr'}
    import os
    ext = os.path.splitext(filename)[1].lower()
    return ext in supported_formats


def safe_json_parse(data: Any, default: Any = None) -> Any:
    """
    安全解析JSON数据
    
    Args:
        data: 待解析的数据（可能是字符串、字典等）
        default: 解析失败时的默认值
    
    Returns:
        解析后的数据或默认值
    """
    import json
    if isinstance(data, dict):
        return data
    if isinstance(data, str):
        try:
            return json.loads(data)
        except json.JSONDecodeError:
            return default
    return default


def truncate_text(text: str, max_length: int = 2000) -> str:
    """
    截断文本，保留最大长度
    
    Args:
        text: 原始文本
        max_length: 最大长度
    
    Returns:
        截断后的文本
    """
    if not text or len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    将文本切分成块，支持重叠
    
    Args:
        text: 原始文本
        chunk_size: 块大小
        overlap: 重叠大小
    
    Returns:
        文本块列表
    """
    if not text:
        return []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

