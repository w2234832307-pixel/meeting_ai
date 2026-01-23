"""
BGE-M3本地Embedding服务
智源研究院开源的多语言向量化模型
"""
from typing import List, Optional
import numpy as np

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import VectorServiceException
from app.core.utils import retry_with_backoff

# BGE-M3相关导入（延迟导入，避免未安装时报错）
try:
    from FlagEmbedding import BGEM3FlagModel
    BGE_M3_AVAILABLE = True
except ImportError:
    BGE_M3_AVAILABLE = False
    logger.warning("⚠️ FlagEmbedding 未安装，无法使用BGE-M3本地Embedding服务")


class BGEM3EmbeddingService:
    """BGE-M3本地Embedding服务类"""
    
    def __init__(self):
        """初始化BGE-M3模型"""
        if not BGE_M3_AVAILABLE:
            raise VectorServiceException(
                "FlagEmbedding未安装，请先安装: pip install FlagEmbedding"
            )
        
        try:
            logger.info("🚀 正在加载BGE-M3模型...")
            
            # 加载BGE-M3模型
            self.model = BGEM3FlagModel(
                settings.BGE_M3_MODEL_NAME,
                use_fp16=settings.BGE_M3_USE_FP16,  # 使用半精度可以节省显存
                device=settings.BGE_M3_DEVICE  # cpu / cuda
            )
            
            # BGE-M3的维度是1024
            self.dim = 1024
            
            logger.info(f"✅ BGE-M3模型加载成功 (设备: {settings.BGE_M3_DEVICE}, 维度: {self.dim})")
            
        except Exception as e:
            logger.error(f"❌ BGE-M3模型加载失败: {e}")
            raise VectorServiceException(f"BGE-M3模型加载失败: {str(e)}")
    
    @retry_with_backoff(max_attempts=2, initial_wait=1.0, max_wait=3.0)
    def get_embedding(self, text: str) -> List[float]:
        """
        获取文本向量（使用dense embedding）
        
        Args:
            text: 文本内容
        
        Returns:
            向量列表（1024维）
        
        Raises:
            VectorServiceException: 向量化异常
        """
        if not text or not text.strip():
            logger.warning("⚠️ 输入文本为空")
            return []
        
        try:
            # 截断文本到最大长度（BGE-M3支持最长8192 tokens）
            max_length = settings.BGE_M3_MAX_LENGTH
            if len(text) > max_length:
                logger.warning(f"⚠️ 文本过长({len(text)}字符)，截断到{max_length}字符")
                text = text[:max_length]
            
            # BGE-M3的encode方法返回字典，包含dense_vecs, sparse_vecs, colbert_vecs
            # 这里使用dense_vecs（密集向量）
            embeddings = self.model.encode(
                [text],  # encode接受列表
                batch_size=settings.BGE_M3_BATCH_SIZE,
                max_length=settings.BGE_M3_MAX_LENGTH,
            )
            
            # 提取dense向量
            if isinstance(embeddings, dict) and 'dense_vecs' in embeddings:
                dense_vec = embeddings['dense_vecs'][0]
            else:
                # 如果直接返回数组
                dense_vec = embeddings[0] if isinstance(embeddings, (list, np.ndarray)) else embeddings
            
            # 转换为Python list
            if isinstance(dense_vec, np.ndarray):
                dense_vec = dense_vec.tolist()
            
            # 验证维度
            if len(dense_vec) != self.dim:
                logger.warning(f"⚠️ 向量维度不匹配: 期望{self.dim}，实际{len(dense_vec)}")
            
            return dense_vec
            
        except Exception as e:
            logger.error(f"❌ BGE-M3向量化失败: {e}")
            raise VectorServiceException(f"BGE-M3向量化失败: {str(e)}")
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        批量获取文本向量（提升效率）
        
        Args:
            texts: 文本列表
        
        Returns:
            向量列表
        """
        if not texts:
            return []
        
        try:
            # 过滤空文本
            valid_texts = [t for t in texts if t and t.strip()]
            if not valid_texts:
                return []
            
            # 批量编码
            embeddings = self.model.encode(
                valid_texts,
                batch_size=settings.BGE_M3_BATCH_SIZE,
                max_length=settings.BGE_M3_MAX_LENGTH,
            )
            
            # 提取dense向量
            if isinstance(embeddings, dict) and 'dense_vecs' in embeddings:
                dense_vecs = embeddings['dense_vecs']
            else:
                dense_vecs = embeddings
            
            # 转换为Python list
            result = []
            for vec in dense_vecs:
                if isinstance(vec, np.ndarray):
                    result.append(vec.tolist())
                else:
                    result.append(vec)
            
            logger.info(f"✅ 批量向量化完成: {len(result)}条文本")
            return result
            
        except Exception as e:
            logger.error(f"❌ 批量向量化失败: {e}")
            raise VectorServiceException(f"批量向量化失败: {str(e)}")


# 创建单例实例（延迟初始化）
_bge_m3_service_instance = None

def get_bge_m3_service():
    """获取BGE-M3服务单例"""
    global _bge_m3_service_instance
    if _bge_m3_service_instance is None:
        if not BGE_M3_AVAILABLE:
            raise VectorServiceException("FlagEmbedding未安装或不可用")
        _bge_m3_service_instance = BGEM3EmbeddingService()
    return _bge_m3_service_instance
