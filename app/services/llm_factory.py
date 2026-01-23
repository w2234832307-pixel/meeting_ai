"""
LLM服务工厂
根据配置动态选择LLM服务（DeepSeek API / 本地Qwen3-14b）
"""
from typing import Protocol

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import LLMServiceException


class LLMServiceProtocol(Protocol):
    """LLM服务协议（接口定义）"""
    
    def judge_rag(self, raw_text: str, template_id: str) -> dict:
        """判断是否需要RAG"""
        ...
    
    def generate_markdown(self, raw_text: str, context: str = "", template_id: str = "default") -> str:
        """生成结构化数据"""
        ...


class LLMServiceFactory:
    """LLM服务工厂类"""
    
    _instance = None
    _current_service = None
    
    @classmethod
    def get_service(cls) -> LLMServiceProtocol:
        """
        获取LLM服务实例（根据配置）
        
        Returns:
            LLM服务实例
        
        Raises:
            LLMServiceException: 服务初始化失败
        """
        # 根据配置选择服务类型
        llm_type = settings.LLM_SERVICE_TYPE.lower()
        
        logger.info(f"🔧 LLM服务类型: {llm_type}")
        
        if llm_type == "api":
            return cls._get_api_llm()
        elif llm_type == "local":
            return cls._get_local_llm()
        else:
            raise LLMServiceException(
                f"不支持的LLM服务类型: {llm_type}，请选择 'api' 或 'local'"
            )
    
    @classmethod
    def _get_api_llm(cls, use_singleton: bool = True):
        """
        获取API LLM服务（DeepSeek等）
        
        Args:
            use_singleton: 是否使用单例（True=使用缓存的实例，False=创建新实例）
        """
        try:
            if use_singleton:
                # 使用启动时初始化的单例
                from app.services.llm import llm_service
                logger.info("✅ 使用API LLM服务 (单例模式)")
                return llm_service
            else:
                # 动态创建新实例
                from app.services.llm import LLMService
                logger.info("✅ 创建新的API LLM服务实例")
                return LLMService()
            
        except Exception as e:
            logger.error(f"❌ API LLM服务初始化失败: {e}")
            raise LLMServiceException(f"API LLM初始化失败: {str(e)}")
    
    @classmethod
    def _get_local_llm(cls, use_singleton: bool = True):
        """
        获取本地LLM服务（Qwen3-14b等）
        
        Args:
            use_singleton: 是否使用单例（True=使用缓存的实例，False=创建新实例）
        """
        try:
            if use_singleton:
                # 使用单例模式
                from app.services.local_llm import get_local_llm_service
                service = get_local_llm_service()
                logger.info("✅ 使用本地LLM服务 (单例模式)")
                return service
            else:
                # 动态创建新实例
                from app.services.local_llm import LocalLLMService
                logger.info("✅ 创建新的本地LLM服务实例")
                return LocalLLMService(test_on_init=False)  # 动态创建时不测试连接
            
        except Exception as e:
            logger.error(f"❌ 本地LLM服务初始化失败: {e}")
            raise LLMServiceException(f"本地LLM初始化失败: {str(e)}")


def get_llm_service() -> LLMServiceProtocol:
    """
    获取LLM服务（便捷函数）
    
    Returns:
        LLM服务实例
    """
    return LLMServiceFactory.get_service()


def get_llm_service_by_name(model_name: str = "auto") -> LLMServiceProtocol:
    """
    根据模型名称动态获取LLM服务
    
    支持的模型：
    - auto: 自动选择（使用配置文件的默认模型）
    - deepseek: DeepSeek API
    - qwen3: 本地 Qwen3-14b
    - api: 使用API模式（兼容）
    - local: 使用本地模式（兼容）
    
    Args:
        model_name: 模型名称
    
    Returns:
        LLM服务实例
    
    Raises:
        LLMServiceException: 服务初始化失败
    """
    # 模型映射
    model_mapping = {
        "deepseek": "api",
        "qwen3": "local",
        "api": "api",
        "local": "local"
    }
    
    # auto 模式：使用配置文件的默认设置（使用单例）
    if model_name == "auto" or model_name not in model_mapping:
        if model_name != "auto":
            logger.warning(f"⚠️ 未知模型 {model_name}，使用默认配置")
        return LLMServiceFactory.get_service()
    
    # 获取服务类型
    service_type = model_mapping[model_name]
    
    logger.info(f"🎯 动态选择模型: {model_name} (类型: {service_type})")
    
    # 判断是否需要创建新实例
    # 如果请求的类型与当前配置不同，创建新实例；否则使用单例
    use_singleton = (service_type == settings.LLM_SERVICE_TYPE.lower())
    
    if not use_singleton:
        logger.info(f"⚡ 动态切换模式: {settings.LLM_SERVICE_TYPE} -> {service_type}")
    
    # 获取服务实例
    if service_type == "api":
        service = LLMServiceFactory._get_api_llm(use_singleton=use_singleton)
    else:
        service = LLMServiceFactory._get_local_llm(use_singleton=use_singleton)
    
    return service
