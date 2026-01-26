import logging
import chromadb
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from app.core.config import settings

# 设置日志
logger = logging.getLogger(__name__)

class VoiceService:
    def __init__(self):
        # ====================================================
        # 1. 自动判断设备 (优先用显卡)
        # ====================================================
        # 如果 settings 里配置了就用 settings 的，没配置就自动检测
        if hasattr(settings, 'FUNASR_DEVICE') and settings.FUNASR_DEVICE:
            self.device = settings.FUNASR_DEVICE
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        logger.info(f"🎙️ 正在初始化 VoiceService... (使用设备: {self.device})")

        # ====================================================
        # 2. 加载声纹提取模型 (Cam++)
        # ====================================================
        try:
            logger.info("📦 开始加载 Cam++ 声纹模型...")
            self.embedding_model = pipeline(
                task=Tasks.speaker_verification,
                model='iic/speech_campplus_sv_zh-cn_16k-common',
                model_revision='v1.0.0',
                device=self.device  # ✅ 这里动态使用检测到的设备
            )
            logger.info("✅ 声纹模型加载成功！")
        except Exception as e:
            logger.critical(f"❌ 声纹模型加载失败，服务将不可用: {e}")
            raise e

        # ====================================================
        # 3. 连接 Chroma 数据库
        # ====================================================
        logger.info(f"🔌 正在连接远程 Chroma: {settings.CHROMA_HOST}:{settings.CHROMA_PORT}")
        try:
            self.client = chromadb.HttpClient(
                host=settings.CHROMA_HOST, 
                port=settings.CHROMA_PORT
            )
            
            # 获取或创建集合
            # Cam++ 输出的是 192 维向量，这里不用手动指定维度，Chroma 会自动处理，
            self.collection = self.client.get_or_create_collection(
                name=settings.CHROMA_COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ 成功连接 Chroma 集合: {settings.CHROMA_COLLECTION_NAME}")
            
        except Exception as e:
            logger.error(f"❌ 连接 Chroma 失败，请检查 .env 配置或网络: {e}")
            # 注意：如果数据库连不上，这里会抛出异常导致程序启动失败
            # 如果你希望程序能继续运行（只是不能存声纹），可以把 raise e 去掉
            raise e

    def extract_vector(self, audio_path: str):
        """
        提取声纹向量
        """
        try:
            # 执行推理
            res = self.embedding_model(audio_path)
            
            # ✅ 增加结果校验，防止模型返回空
            if res and 'spk_embedding' in res:
                vector = res['spk_embedding']
                
                # ✅ 格式转换：确保转成 Python list
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                
                # 打印一下维度（调试用，正式上线可以注释掉）
                # logger.debug(f"📐 提取向量成功，维度: {len(vector)}")
                return vector
            else:
                logger.error(f"❌ 提取失败，模型未返回 embedding: {res}")
                return None
            
        except Exception as e:
            logger.error(f"❌ 提取声纹向量异常: {e}")
            return None
    
    def save_identity(self, employee_id: str, name: str, vector: list):
        """
        保存员工声纹到 Chroma
        """
        try:
            self.collection.add(
                ids=[str(employee_id)],  # 覆盖式更新（同一个工号只存一个声纹）
                embeddings=[vector],
                metadatas=[{
                    "name": name, 
                    "employee_id": str(employee_id),
                    "create_time": "2026-01-XX" # 这里可以加个时间戳
                }]
            )
            logger.info(f"💾 声纹已入库: {name} (工号: {employee_id})")
            return True
        except Exception as e:
            logger.error(f"❌ 声纹入库失败: {e}")
            raise e


try:
    # 注意：这意味着这行代码一运行（比如 import 这个文件时），就会开始加载模型
    voice_service = VoiceService()
except Exception as e:
    logger.error(f"⚠️ VoiceService 初始化失败: {e}")
    voice_service = None

# 如果你需要 FastAPI 的依赖注入，可以用这个函数
def get_voice_service():
    if voice_service is None:
        raise RuntimeError("VoiceService 未成功初始化，请检查日志")
    return voice_service