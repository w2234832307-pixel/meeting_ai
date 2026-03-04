"""
向量检索服务
使用 Chroma 存储向量，支持多种 Embedding 服务
"""
import json
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import VectorServiceException
from app.core.utils import chunk_text
from app.services.embedding_factory import get_embedding_service


class VectorService:
    """向量检索服务类（基于 Chroma）"""
    
    def __init__(self):
        """初始化向量服务"""
        # 使用知识库表名（会议纪要），与声纹库分离
        self.collection_name = settings.CHROMA_KNOWLEDGE_COLLECTION_NAME
        
        # 获取Embedding服务（根据配置自动选择）
        try:
            self.embedding_service = get_embedding_service()
            self.dim = getattr(self.embedding_service, 'dim', 1024)  # BGE-M3默认1024
            logger.info(f"✅ Embedding服务初始化成功，向量维度: {self.dim}")
        except Exception as e:
            logger.error(f"❌ Embedding服务初始化失败: {e}")
            self.embedding_service = None
            self.dim = 1024  # 默认维度
        
        self.collection = None
        self.client = None
        
        # 连接Chroma（允许降级运行）
        try:
            self._connect_chroma()
            self._init_collection()
        except Exception as e:
            logger.error(f"❌ Chroma初始化失败，服务将以降级模式运行: {e}")
            # 不抛出异常，允许服务在其他功能正常时继续运行
    
    def _connect_chroma(self) -> None:
        """连接 Chroma"""
        try:
            # 连接到远程 Chroma 服务器
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=False
                )
            )
            
            # 测试连接
            self.client.heartbeat()
            
            logger.info(f"🔌 Chroma连接成功: {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
        except Exception as e:
            logger.error(f"❌ Chroma连接失败: {e}")
            raise VectorServiceException(f"Chroma连接失败: {str(e)}")
    
    def _init_collection(self) -> None:
        """初始化 Chroma 集合"""
        try:
            # 尝试获取已存在的集合
            try:
                self.collection = self.client.get_collection(
                    name=self.collection_name,
                    embedding_function=None  # 我们自己管理 embedding
                )
                logger.info(f"✅ 集合 {self.collection_name} 已存在，已加载")
            except Exception:
                # 集合不存在，创建新集合
                logger.info(f"✨ 集合 {self.collection_name} 不存在，正在创建...")
                self.collection = self.client.create_collection(
                    name=self.collection_name,
                    embedding_function=None,  # 我们自己管理 embedding
                    metadata={"description": "员工心声知识库"}
                )
                logger.info(f"✅ 集合 {self.collection_name} 创建完成")
            
        except Exception as e:
            logger.error(f"❌ 集合初始化失败: {e}")
            raise VectorServiceException(f"集合初始化失败: {str(e)}")
    
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
        
        try:
            return self.embedding_service.get_embedding(text)
        except Exception as e:
            logger.error(f"❌ 向量化失败: {e}")
            return []
    
    def search_similar(self, query_text: str, top_k: int = 3, min_score: float = 0.7) -> str:
        """
        搜索相似的历史片段
        
        Args:
            query_text: 查询文本
            top_k: 返回最相似的前k个结果
            min_score: 最小相似度阈值（余弦相似度，0-1之间，值越大越相似）
        
        Returns:
            拼接的相关文本
        """
        if not query_text:
            return ""
        
        if not self.collection:
            logger.warning("⚠️ Chroma集合未初始化，无法进行向量检索")
            return ""
        
        try:
            # 1. 将查询文本转为向量
            query_vec = self.get_embedding(query_text)
            if not query_vec:
                logger.warning("⚠️ 查询文本向量化失败")
                return ""
            
            # 2. 在Chroma中搜索
            results = self.collection.query(
                query_embeddings=[query_vec],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # 3. 拼接结果
            context_list = []
            
            if results and results.get("documents"):
                documents = results["documents"][0]  # 第一个查询的结果
                metadatas = results.get("metadatas", [[]])[0]
                distances = results.get("distances", [[]])[0]
                
                for i, doc in enumerate(documents):
                    # Chroma 使用 L2 距离（越小越相似）
                    # 转换为相似度分数（0-1之间）
                    distance = distances[i] if i < len(distances) else float('inf')
                    
                    # 简单的距离到相似度转换：similarity = 1 / (1 + distance)
                    # 对于余弦距离，可以直接用 1 - distance（如果Chroma配置了余弦）
                    similarity = 1 / (1 + distance)
                    
                    # 过滤相似度太低的结果
                    if similarity < min_score:
                        continue
                    
                    metadata = metadatas[i] if i < len(metadatas) else {}
                    
                    if doc:
                        context_list.append(f"- 相关记录（相似度: {similarity:.2f}）: {doc}")
            
            logger.info(f"🔍 检索到 {len(context_list)} 条相关历史")
            return "\n".join(context_list)
            
        except Exception as e:
            logger.error(f"❌ 搜索异常: {e}")
            return ""
    
    def save_knowledge(
        self, 
        text: str, 
        source_id: int, 
        extra_meta: Optional[Dict[str, Any]] = None,
        chunk_size: int = 500,
        overlap: int = 50
    ) -> int:
        """
        归档功能：将最终的会议纪要切片存入 Chroma
        
        Args:
            text: 文本内容
            source_id: 对应MySQL中的minutes_draft_id
            extra_meta: 其他元数据（如user_id）
            chunk_size: 切片大小
            overlap: 重叠大小
        
        Returns:
            成功保存的切片数量
        """
        if not self.collection:
            logger.warning("⚠️ Chroma集合未初始化，无法保存知识")
            return 0
        
        if not text or not text.strip():
            logger.warning("⚠️ 文本内容为空，跳过保存")
            return 0
        
        try:
            # 切片
            chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
            
            if not chunks:
                logger.warning("⚠️ 切片后为空，跳过保存")
                return 0
            
            saved_count = 0
            
            # 批量处理切片
            ids_batch = []
            embeddings_batch = []
            documents_batch = []
            metadatas_batch = []
            
            for i, chunk in enumerate(chunks):
                if not chunk.strip():
                    continue
                
                # 获取向量
                vec = self.get_embedding(chunk)
                if not vec:
                    logger.warning(f"⚠️ 切片 {i+1} 向量化失败，跳过")
                    continue
                
                # 构造元数据
                meta_dict = {
                    "source_id": source_id,
                    "chunk_index": i
                }
                if extra_meta:
                    meta_dict.update(extra_meta)
                
                # 生成唯一ID：source_id + chunk_index
                chunk_id = f"{source_id}_{i}"
                
                ids_batch.append(chunk_id)
                embeddings_batch.append(vec)
                documents_batch.append(chunk)
                metadatas_batch.append(meta_dict)
            
            # 批量插入到 Chroma
            if embeddings_batch:
                self.collection.add(
                    ids=ids_batch,
                    embeddings=embeddings_batch,
                    documents=documents_batch,
                    metadatas=metadatas_batch
                )
                
                saved_count = len(embeddings_batch)
                logger.info(f"💾 已存储 {saved_count} 个知识切片 (SourceID: {source_id})")
            
            return saved_count
            
        except Exception as e:
            logger.error(f"❌ 存储失败: {e}")
            raise VectorServiceException(f"存储失败: {str(e)}")
    
    def is_available(self) -> bool:
        """检查服务是否可用"""
        return self.collection is not None


# 创建单例实例
try:
    vector_service = VectorService()
except Exception as e:
    logger.error(f"❌ 向量服务初始化失败: {e}")
    vector_service = None
