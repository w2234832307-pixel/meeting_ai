"""
Embedding服务工厂
根据配置动态选择Embedding服务（腾讯云 / OpenAI / BGE-M3本地）
"""
from typing import Protocol, List

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import VectorServiceException


class EmbeddingServiceProtocol(Protocol):
    """Embedding服务协议（接口定义）"""
    
    dim: int  # 向量维度
    
    def get_embedding(self, text: str) -> List[float]:
        """获取文本向量"""
        ...


class EmbeddingServiceFactory:
    """Embedding服务工厂类"""
    
    _instance = None
    _current_service = None
    
    @classmethod
    def get_service(cls) -> EmbeddingServiceProtocol:
        """
        获取Embedding服务实例（根据配置）
        
        Returns:
            Embedding服务实例
        
        Raises:
            VectorServiceException: 服务初始化失败
        """
        # 根据配置选择服务类型
        embedding_type = settings.EMBEDDING_SERVICE.lower()
        
        logger.info(f"🔧 Embedding服务类型: {embedding_type}")
        
        if embedding_type == "bge-m3":
            return cls._get_bge_m3()
        elif embedding_type == "tencent":
            return cls._get_tencent()
        elif embedding_type == "openai":
            return cls._get_openai()
        else:
            logger.warning(f"⚠️ 未知的Embedding服务类型: {embedding_type}，使用BGE-M3作为默认")
            return cls._get_bge_m3()
    
    @classmethod
    def _get_bge_m3(cls):
        """获取BGE-M3本地服务（推荐）"""
        try:
            from app.services.bge_m3_embedding import get_bge_m3_service
            
            service = get_bge_m3_service()
            logger.info("✅ 使用BGE-M3本地Embedding服务")
            return service
            
        except Exception as e:
            logger.error(f"❌ BGE-M3 Embedding服务初始化失败: {e}")
            raise VectorServiceException(f"BGE-M3初始化失败: {str(e)}")
    
    @classmethod
    def _get_tencent(cls):
        """获取腾讯云Embedding服务"""
        try:
            from app.services.tencent_embedding import TencentEmbeddingService
            
            service = TencentEmbeddingService()
            logger.info("✅ 使用腾讯云Embedding服务")
            return service
            
        except Exception as e:
            logger.error(f"❌ 腾讯云Embedding服务初始化失败: {e}")
            raise VectorServiceException(f"腾讯云Embedding初始化失败: {str(e)}")
    
    @classmethod
    def _get_openai(cls):
        """获取OpenAI兼容Embedding服务"""
        try:
            from app.services.tencent_embedding import OpenAICompatibleEmbeddingService
            
            service = OpenAICompatibleEmbeddingService()
            logger.info("✅ 使用OpenAI兼容Embedding服务")
            return service
            
        except Exception as e:
            logger.error(f"❌ OpenAI兼容Embedding服务初始化失败: {e}")
            raise VectorServiceException(f"OpenAI Embedding初始化失败: {str(e)}")


def get_embedding_service() -> EmbeddingServiceProtocol:
    """
    获取Embedding服务（便捷函数）
    
    Returns:
        Embedding服务实例
    """
    return EmbeddingServiceFactory.get_service()
