import logging
import chromadb
import torch
import tempfile
from typing import List, Dict, Tuple, Optional


def _fix_datasets_compatibility():
    """修复 datasets 与 modelscope 的兼容性问题（LargeList 等）"""
    try:
        import datasets

        # 修复 LargeList（部分新版本 datasets 中已移除）
        if not hasattr(datasets, "LargeList"):
            try:
                from datasets import LargeList  # 尝试直接导入（旧版本）
            except ImportError:
                try:
                    import pyarrow as pa

                    if hasattr(pa, "large_list"):
                        datasets.LargeList = pa.large_list
                    elif hasattr(pa, "LargeList"):
                        datasets.LargeList = pa.LargeList
                except Exception:
                    # 如果 pyarrow 也没有对应实现，就静默跳过，让后续代码自行处理
                    pass

        # 修复 _FEATURE_TYPES（datasets 2.19+ 中可能位置变化或被移除）
        try:
            from datasets.features.features import _FEATURE_TYPES  # 旧位置
        except ImportError:
            try:
                from datasets.features import _FEATURE_TYPES  # 尝试新位置
            except ImportError:
                try:
                    import datasets.features.features as features_module

                    if not hasattr(features_module, "_FEATURE_TYPES"):
                        # 创建一个空占位符，避免 modelscope 导入时报错
                        features_module._FEATURE_TYPES = {}
                except Exception:
                    pass
    except Exception:
        # 如果 datasets 自身都导入失败，保持原状，由上层捕获错误
        pass


# 必须在导入 modelscope 之前执行修复
_fix_datasets_compatibility()

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
    
    @property
    def enabled(self) -> bool:
        """检查声纹服务是否启用（声纹库是否为空）"""
        try:
            count = self.collection.count()
            return count > 0
        except:
            return False
    
    def extract_speaker_segments(self, audio_path: str, transcript: List[Dict], duration: int = 10) -> Dict[str, str]:
        """
        为每个说话人提取音频片段（每个speaker_id只提取一次）
        
        Args:
            audio_path: 原始音频文件路径
            transcript: ASR识别结果，包含speaker_id和时间戳
            duration: 提取音频时长（秒）
        
        Returns:
            {speaker_id: audio_segment_path}
        """
        import subprocess
        import tempfile
        from pathlib import Path
        
        if not self.enabled:
            return {}
        
        speaker_segments = {}
        speaker_times = {}  # {speaker_id: [(start, end), ...]}
        
        # 1. 收集每个说话人的所有时间段
        for item in transcript:
            speaker_id = item.get("speaker_id", "unknown")
            start_time = item.get("start_time", 0)
            end_time = item.get("end_time", 0)
            
            if speaker_id not in speaker_times:
                speaker_times[speaker_id] = []
            
            speaker_times[speaker_id].append((start_time, end_time))
        
        # 2. 为每个说话人提取音频片段（每个speaker_id只提取一次）
        logger.info(f"🔍 开始为 {len(speaker_times)} 个不同的speaker_id提取音频片段（每个ID只提取一次）")
        for speaker_id, times in speaker_times.items():
            if speaker_id == "unknown":
                continue
            
            try:
                # 找出该说话人最长的连续片段
                sorted_times = sorted(times, key=lambda x: x[1] - x[0], reverse=True)
                
                # 取第一段（最长的）- 每个speaker_id只提取一次
                if sorted_times:
                    start, end = sorted_times[0]
                    segment_end = min(end, start + duration)
                    
                    # 使用ffmpeg提取片段
                    temp_dir = Path(tempfile.gettempdir())
                    output_path = temp_dir / f"speaker_{speaker_id}_{int(start)}.wav"
                    
                    cmd = [
                        "ffmpeg",
                        "-i", audio_path,
                        "-ss", str(start),
                        "-t", str(segment_end - start),
                        "-ac", "1",
                        "-ar", "16000",
                        "-y",
                        "-loglevel", "error",
                        str(output_path)
                    ]
                    
                    subprocess.run(cmd, check=True, capture_output=True, timeout=30)
                    
                    if output_path.exists():
                        speaker_segments[speaker_id] = str(output_path.resolve())
                        logger.info(f"✅ 提取说话人 {speaker_id} 音频片段: {start:.1f}s - {segment_end:.1f}s")
                    
            except Exception as e:
                logger.error(f"❌ 提取说话人 {speaker_id} 音频失败: {e}")
        
        return speaker_segments
    
    def match_speakers(self, speaker_segments: Dict[str, str], threshold: float = 0.75) -> Dict[str, tuple]:
        """
        匹配说话人身份（每个speaker_id只匹配一次）
        
        Args:
            speaker_segments: {speaker_id: audio_path} 或 {speaker_id: [audio_path1, audio_path2, ...]}
            threshold: 相似度阈值（0-1）
        
        Returns:
            {speaker_id: (employee_id, name, similarity)}
        """
        if not self.enabled:
            return {}
        
        matched = {}
        
        logger.info(f"🔍 开始匹配 {len(speaker_segments)} 个不同的speaker_id（每个ID只匹配一次）")
        for speaker_id, audio_data in speaker_segments.items():
            try:
                # 支持单个路径或路径列表（用于计算均值）
                if isinstance(audio_data, str):
                    audio_paths = [audio_data]
                elif isinstance(audio_data, list):
                    audio_paths = audio_data
                else:
                    logger.warning(f"⚠️ 说话人 {speaker_id} 音频数据格式错误: {type(audio_data)}")
                    continue
                
                # 1. 提取所有音频片段的声纹向量
                vectors = []
                for audio_path in audio_paths:
                    vector = self.extract_vector(audio_path)
                    if vector is not None:
                        vectors.append(vector)
                
                if not vectors:
                    logger.warning(f"⚠️ 说话人 {speaker_id} 所有音频片段声纹提取失败")
                    continue
                
                # 2. 计算均值向量（如果多个片段）
                import numpy as np
                if len(vectors) > 1:
                    vectors_array = np.array(vectors)
                    mean_vector = np.mean(vectors_array, axis=0).tolist()
                    logger.info(f"✅ 说话人 {speaker_id}: {len(vectors)} 个片段，已计算均值向量")
                else:
                    mean_vector = vectors[0]
                
                # 3. 在声纹库中搜索
                results = self.collection.query(
                    query_embeddings=[mean_vector],
                    n_results=1
                )
                
                if not results['ids'] or len(results['ids'][0]) == 0:
                    logger.warning(f"⚠️ 说话人 {speaker_id} 未在声纹库中找到匹配")
                    continue
                
                # 4. 获取匹配结果
                employee_id = results['ids'][0][0]
                metadata = results['metadatas'][0][0]
                distance = results['distances'][0][0] if 'distances' in results else 0.5
                
                # 距离转相似度（cosine距离: 0=完全相同, 2=完全相反）
                similarity = 1 - (distance / 2.0)
                
                name = metadata.get('name', '未知')
                
                if similarity >= threshold:
                    matched[speaker_id] = (employee_id, name, similarity)
                    logger.info(f"✅ 说话人 {speaker_id} 匹配成功: {name} (相似度: {similarity:.2%})")
                else:
                    logger.warning(f"⚠️ 说话人 {speaker_id} 相似度过低: {similarity:.2%} < {threshold:.2%}")
                
                # 清理临时文件
                for audio_path in audio_paths:
                    try:
                        import os
                        if os.path.exists(audio_path) and audio_path.startswith(tempfile.gettempdir()):
                            os.remove(audio_path)
                    except:
                        pass
                
            except Exception as e:
                logger.error(f"❌ 匹配说话人 {speaker_id} 失败: {e}")
        
        return matched
    
    def replace_speaker_ids(self, transcript: List[Dict], matched: Dict[str, tuple]) -> List[Dict]:
        """
        将speaker_id替换为真实姓名
        
        Args:
            transcript: ASR识别结果
            matched: 匹配结果 {speaker_id: (employee_id, name, similarity)}
        
        Returns:
            更新后的transcript
        """
        for item in transcript:
            speaker_id = item.get("speaker_id")
            if speaker_id in matched:
                employee_id, name, similarity = matched[speaker_id]
                item['speaker_id'] = name
                item['employee_id'] = employee_id
                item['similarity'] = similarity
        
        return transcript


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