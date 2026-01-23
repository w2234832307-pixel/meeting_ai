import logging
import chromadb
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from app.core.config import settings

logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        # 1. 加载声纹提取模型 (Cam++)
        # 这个模型在你本地跑，用你的显卡/CPU
        logger.info("🎙️ 正在加载声纹模型 (Cam++)...")
        self.embedding_model = pipeline(
            task=Tasks.speaker_verification,
            model='iic/speech_campplus_sv_zh-cn_16k-common',
            model_revision='v1.0.0',
            device="cpu" # 或者 settings.FUNASR_DEVICE
        )

        # 2. 连接同事的 Chroma 数据库
        logger.info(f"🔌 正在连接远程 Chroma: {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
        try:
            # 🔥 关键：使用 HttpClient 连接远程服务
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST, 
                port=settings.CHROMA_PORT
            )
            
            # 获取或创建集合
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("✅ 成功连接到 Chroma 数据库！")
            
        except Exception as e:
            logger.error(f"❌ 连接 Chroma 失败，请检查 IP 和防火墙设置: {e}")
            raise e

    def extract_vector(self, audio_path: str):
        """
        提取声纹向量
        """
        try:
            res = self.embedding_model(audio_path)
            if 'spk_embedding' not in res:
                logger.error("❌ 无法从音频中提取声纹特征")
                return None
            
            vector = res['spk_embedding'].tolist()
            logger.info(f"✅ 成功提取声纹向量 (维度: {len(vector)})")
            return vector
        except Exception as e:
            logger.error(f"❌ 提取声纹向量失败: {e}")
            return None
    
    def save_identity(self, employee_id: str, name: str, vector: list):
        """
        保存员工声纹到 Chroma
        """
        try:
            self.collection.add(
                ids=[str(employee_id)],  # 覆盖式更新
                embeddings=[vector],
                metadatas=[{
                    "name": name, 
                    "employee_id": str(employee_id)
                }]
            )
            logger.info(f"✅ 成功保存声纹: {name} (工号: {employee_id})")
            return True
        except Exception as e:
            logger.error(f"❌ 保存声纹失败: {e}")
            raise e

# 单例导出（延迟实例化，避免启动时加载）
_voice_service_instance = None

def get_voice_service():
    """延迟实例化，避免启动时加载"""
    global _voice_service_instance
    if _voice_service_instance is None:
        _voice_service_instance = VoiceService()
    return _voice_service_instance

# 为了兼容旧代码，保留 voice_service 变量（但它会在首次导入时实例化）
voice_service = VoiceService()