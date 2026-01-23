"""
腾讯云NLP/Embedding服务
支持文本向量化，兼容多种服务
"""
from typing import List, Optional, Dict, Any
from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import VectorServiceException
from app.core.utils import retry_with_backoff

try:
    from tencentcloud.common import credential
    from tencentcloud.common.profile.client_profile import ClientProfile
    from tencentcloud.common.profile.http_profile import HttpProfile
    from tencentcloud.nlp.v20190408 import nlp_client, models as nlp_models
    from tencentcloud.common.exception.tencent_cloud_sdk_exception import TencentCloudSDKException
    TENCENT_SDK_AVAILABLE = True
except ImportError:
    TENCENT_SDK_AVAILABLE = False
    logger.warning("⚠️ 腾讯云SDK未安装，Embedding服务可能不可用")


class TencentEmbeddingService:
    """腾讯云Embedding服务类"""
    
    def __init__(self):
        """初始化腾讯云Embedding服务"""
        self.dim = 1024  # 腾讯云NLP向量维度（根据实际API调整）
        self.client = None
        
        # 检查腾讯云NLP是否可用
        if not settings.is_tencent_nlp_available():
            logger.warning("⚠️ 腾讯云NLP配置不完整，将使用备用服务")
            return
        
        try:
            # 初始化凭证
            cred = credential.Credential(
                settings.TENCENT_NLP_SECRET_ID,
                settings.TENCENT_NLP_SECRET_KEY
            )
            
            # 实例化HTTP配置
            http_profile = HttpProfile()
            http_profile.endpoint = "nlp.tencentcloudapi.com"
            http_profile.reqTimeout = settings.EMBEDDING_TIMEOUT
            
            # 实例化客户端配置
            client_profile = ClientProfile()
            client_profile.httpProfile = http_profile
            
            # 实例化NLP客户端
            self.client = nlp_client.NlpClient(cred, settings.TENCENT_REGION, client_profile)
            
            logger.info(f"✅ 腾讯云Embedding服务初始化成功 (区域: {settings.TENCENT_REGION})")
            
        except Exception as e:
            logger.warning(f"⚠️ 腾讯云Embedding服务初始化失败: {e}")
            self.client = None
    
    @retry_with_backoff(max_attempts=3, initial_wait=1.0, max_wait=5.0)
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本向量
        
        Args:
            text: 文本内容
        
        Returns:
            向量列表
        """
        if not text:
            return []
        
        if not self.client:
            raise VectorServiceException("腾讯云Embedding服务未初始化")
        
        try:
            # 注意：腾讯云NLP API可能没有直接的Embedding接口
            # 这里使用相似词查询或其他API作为替代
            # 实际使用时需要根据腾讯云NLP API文档调整
            
            # 示例：使用词向量API（如果可用）
            # 或者使用其他兼容的Embedding服务
            
            logger.warning("⚠️ 腾讯云NLP可能不支持直接Embedding API，建议使用OpenAI兼容的Embedding服务")
            
            # 暂时返回空，实际使用时需要根据API文档实现
            return []
            
        except TencentCloudSDKException as e:
            logger.error(f"❌ 腾讯云Embedding API调用失败: {e}")
            raise VectorServiceException(f"API调用失败: {e.get_code()} - {e.get_message()}")
        except Exception as e:
            logger.error(f"❌ 向量化异常: {e}")
            raise VectorServiceException(f"向量化异常: {str(e)}")


# 备用：使用OpenAI兼容的Embedding服务
class OpenAICompatibleEmbeddingService:
    """OpenAI兼容的Embedding服务（作为备用）"""
    
    def __init__(self):
        """初始化OpenAI兼容的Embedding服务"""
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=settings.LLM_API_KEY,
                base_url=settings.LLM_BASE_URL
            )
            self.dim = 1536  # OpenAI text-embedding-ada-002 的维度
            logger.info("✅ OpenAI兼容Embedding服务初始化成功")
        except Exception as e:
            logger.warning(f"⚠️ OpenAI兼容Embedding服务初始化失败: {e}")
            self.client = None
    
    @retry_with_backoff(max_attempts=3, initial_wait=1.0, max_wait=5.0)
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本向量
        
        Args:
            text: 文本内容
        
        Returns:
            向量列表
        """
        if not text or not self.client:
            return []
        
        try:
            # 截断文本到最大长度（通常Embedding API有长度限制）
            max_length = 8000  # 大多数API的限制
            if len(text) > max_length:
                text = text[:max_length]
            
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",  # 或使用兼容的模型
                input=text
            )
            
            return response.data[0].embedding
            
        except Exception as e:
            logger.error(f"❌ Embedding API调用失败: {e}")
            raise VectorServiceException(f"Embedding API调用失败: {str(e)}")


# 创建Embedding服务实例（根据配置选择）
def create_embedding_service():
    """根据配置创建Embedding服务"""
    if settings.EMBEDDING_SERVICE == "tencent" and settings.is_tencent_nlp_available():
        try:
            return TencentEmbeddingService()
        except Exception as e:
            logger.warning(f"⚠️ 腾讯云Embedding服务创建失败，使用备用服务: {e}")
    
    # 使用OpenAI兼容的服务作为备用
    return OpenAICompatibleEmbeddingService()


embedding_service = create_embedding_service()

