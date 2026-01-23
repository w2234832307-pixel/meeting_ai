"""
日志模块 - 避免循环导入
"""
import sys
import logging
from pathlib import Path
from logging.handlers import RotatingFileHandler

def _get_log_dir():
    """获取日志目录，避免导入settings"""
    base_dir = Path(__file__).resolve().parent.parent.parent
    return base_dir / "logs"

def setup_logger(name: str = "meeting_ai", log_dir: Path = None):
    """
    配置全局 Logger
    
    Args:
        name: logger名称
        log_dir: 日志目录，如果不提供则自动获取
    """
    # 1. 创建 logger 对象
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 如果已经有 handler  了，就别加了（防止重复打印）
    if logger.handlers:
        return logger

    # 2. 定义格式 (时间 - 级别 - 文件名:行号 - 消息)
    formatter = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    # 3. Handler A: 输出到控制台 (Console)
    # 使用 sys.stderr 而不是 sys.stdout，避免被 uvicorn 缓冲
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)  # ✅ 明确设置级别
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # ✅ 禁用日志传播，避免重复输出
    logger.propagate = False

    # 4. Handler B: 输出到文件 (File) -> logs/app.log
    # maxBytes=10MB, backupCount=5 (保留最近5个文件)
    if log_dir is None:
        log_dir = _get_log_dir()
    
    # 确保日志目录存在
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / "app.log"
    file_handler = RotatingFileHandler(
        log_file, maxBytes=10*1024*1024, backupCount=5, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

# 初始化一个全局 logger 供其他文件直接 import 使用
logger = setup_logger()