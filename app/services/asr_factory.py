"""
ASR服务工厂
根据配置动态选择ASR服务（腾讯云 / FunASR本地）
"""
from typing import Protocol, Dict, Any, List

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import ASRServiceException


class ASRServiceProtocol(Protocol):
    """ASR服务协议（接口定义）"""
    
    def transcribe(self, file_path: str) -> Dict[str, Any]:
        """
        识别音频文件
        
        Args:
            file_path: 音频文件路径
        
        Returns:
            包含 text 和 transcript 的字典
        """
        ...


class ASRServiceFactory:
    """ASR服务工厂类"""
    
    _instance = None
    _current_service = None
    
    @classmethod
    def get_service(cls) -> ASRServiceProtocol:
        """
        获取ASR服务实例（根据配置）
        
        Returns:
            ASR服务实例
        
        Raises:
            ASRServiceException: 服务初始化失败
        """
        # 根据配置选择服务类型
        asr_type = settings.ASR_SERVICE_TYPE.lower()
        
        logger.info(f"🔧 ASR服务类型: {asr_type}")
        
        if asr_type == "tencent":
            return cls._get_tencent_asr()
        elif asr_type == "funasr":
            return cls._get_funasr()
        else:
            raise ASRServiceException(
                f"不支持的ASR服务类型: {asr_type}，请选择 'tencent' 或 'funasr'"
            )
    
    @classmethod
    def _get_tencent_asr(cls):
        """获取腾讯云ASR服务"""
        try:
            from app.services.tencent_asr import asr_service
            
            if asr_service is None:
                # 如果单例初始化失败，尝试重新创建
                from app.services.tencent_asr import TencentASRService
                return TencentASRService()
            
            logger.info("✅ 使用腾讯云ASR服务")
            return asr_service
            
        except Exception as e:
            logger.error(f"❌ 腾讯云ASR服务初始化失败: {e}")
            raise ASRServiceException(f"腾讯云ASR初始化失败: {str(e)}")
    
    @classmethod
    def _get_funasr(cls):
        """获取FunASR本地服务"""
        try:
            from app.services.funasr_service import get_funasr_service
            
            service = get_funasr_service()
            logger.info("✅ 使用FunASR本地服务")
            return service
            
        except Exception as e:
            logger.error(f"❌ FunASR本地服务初始化失败: {e}")
            raise ASRServiceException(f"FunASR初始化失败: {str(e)}")


def get_asr_service() -> ASRServiceProtocol:
    """
    获取ASR服务（便捷函数）
    
    Returns:
        ASR服务实例
    """
    return ASRServiceFactory.get_service()


def get_asr_service_by_name(model_name: str = "auto") -> ASRServiceProtocol:
    """
    根据模型名称动态获取ASR服务
    
    支持的模型：
    - auto: 自动选择（使用配置文件的默认模型）
    - tencent: 腾讯云ASR
    - funasr: 本地FunASR
    
    Args:
        model_name: 模型名称
    
    Returns:
        ASR服务实例
    
    Raises:
        ASRServiceException: 服务初始化失败
    """
    # 模型映射
    model_mapping = {
        "tencent": "tencent",
        "funasr": "funasr"
    }
    
    # auto 模式：使用配置文件的默认设置
    if model_name == "auto" or model_name not in model_mapping:
        if model_name != "auto":
            logger.warning(f"⚠️ 未知ASR模型 {model_name}，使用默认配置")
        return ASRServiceFactory.get_service()
    
    # 获取服务类型
    service_type = model_mapping[model_name]
    
    logger.info(f"🎯 动态选择ASR模型: {model_name} (类型: {service_type})")
    
    # 临时修改配置并获取服务
    original_type = settings.ASR_SERVICE_TYPE
    try:
        settings.ASR_SERVICE_TYPE = service_type
        
        if service_type == "tencent":
            service = ASRServiceFactory._get_tencent_asr()
        else:
            service = ASRServiceFactory._get_funasr()
        
        return service
    finally:
        # 恢复原配置
        settings.ASR_SERVICE_TYPE = original_type
