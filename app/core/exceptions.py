"""
自定义异常类
"""


class ConfigurationException(Exception):
    """配置异常"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class ASRServiceException(Exception):
    """ASR服务异常"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class LLMServiceException(Exception):
    """LLM服务异常"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class VectorServiceException(Exception):
    """向量服务异常"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class DocumentServiceException(Exception):
    """文档服务异常"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}


class DownloadServiceException(Exception):
    """下载服务异常"""
    def __init__(self, message: str, details: dict = None):
        super().__init__(message)
        self.details = details or {}
