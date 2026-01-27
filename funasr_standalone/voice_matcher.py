"""
声纹匹配服务
用于将ASR识别的speaker_id映射到真实员工姓名
"""
import logging
import chromadb
import torch
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import tempfile
import os

logger = logging.getLogger(__name__)


class VoiceMatcher:
    """声纹匹配器"""
    
    def __init__(self, 
                 chroma_host: str = "192.168.211.74",
                 chroma_port: int = 8000,
                 collection_name: str = "employee_voice_library",
                 device: str = None):
        """
        初始化声纹匹配器
        
        Args:
            chroma_host: ChromaDB地址
            chroma_port: ChromaDB端口
            collection_name: 声纹库集合名称
            device: 设备（cuda/cpu）
        """
        self.enabled = False
        
        try:
            # 自动检测设备
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device
            
            logger.info(f"🎙️ 正在初始化声纹匹配器... (设备: {self.device})")
            
            # 加载Cam++声纹模型
            logger.info("📦 加载 Cam++ 声纹模型...")
            self.embedding_model = pipeline(
                task=Tasks.speaker_verification,
                model='iic/speech_campplus_sv_zh-cn_16k-common',
                model_revision='v1.0.0',
                device=self.device
            )
            logger.info("✅ 声纹模型加载成功")
            
            # 连接ChromaDB
            logger.info(f"🔌 连接 ChromaDB: {chroma_host}:{chroma_port}")
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
            
            # 获取声纹库集合
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"✅ 连接声纹库成功: {collection_name}")
            
            # 检查声纹库是否为空
            count = self.collection.count()
            if count == 0:
                logger.warning("⚠️ 声纹库为空，声纹识别将被禁用")
                self.enabled = False
            else:
                logger.info(f"✅ 声纹库已就绪，共 {count} 个员工声纹")
                self.enabled = True
            
        except Exception as e:
            logger.error(f"❌ 声纹匹配器初始化失败: {e}")
            logger.warning("⚠️ 声纹识别功能将被禁用，将使用默认speaker_id")
            self.enabled = False
    
    def extract_speaker_segments(self,
                                  audio_path: str,
                                  transcript: List[Dict],
                                  duration: int = 10) -> Dict[str, str]:
        """
        为每个说话人提取音频片段
        
        Args:
            audio_path: 原始音频文件路径
            transcript: ASR识别结果，包含speaker_id和时间戳
            duration: 提取音频时长（秒）
        
        Returns:
            {speaker_id: audio_segment_path}
        """
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
        
        # 2. 为每个说话人提取音频片段
        for speaker_id, times in speaker_times.items():
            if speaker_id == "unknown":
                continue
            
            try:
                # 找出该说话人最长的连续片段
                sorted_times = sorted(times, key=lambda x: x[1] - x[0], reverse=True)
                
                # 取前几段，累计达到指定时长
                accumulated_duration = 0
                selected_segments = []
                
                for start, end in sorted_times:
                    segment_duration = end - start
                    if segment_duration >= 2:  # 至少2秒的片段才考虑
                        selected_segments.append((start, end))
                        accumulated_duration += segment_duration
                        
                        if accumulated_duration >= duration:
                            break
                
                if not selected_segments:
                    logger.warning(f"⚠️ 说话人 {speaker_id} 没有足够长的音频片段")
                    continue
                
                # 提取第一段（最长的）
                start, end = selected_segments[0]
                segment_path = self._extract_audio_segment(
                    audio_path, 
                    start, 
                    min(end, start + duration),
                    speaker_id
                )
                
                if segment_path:
                    speaker_segments[speaker_id] = segment_path
                    logger.info(f"✅ 提取说话人 {speaker_id} 音频: {start:.1f}s - {end:.1f}s")
                
            except Exception as e:
                logger.error(f"❌ 提取说话人 {speaker_id} 音频失败: {e}")
        
        return speaker_segments
    
    def _extract_audio_segment(self,
                                audio_path: str,
                                start_time: float,
                                end_time: float,
                                speaker_id: str) -> Optional[str]:
        """
        使用ffmpeg提取音频片段
        
        Args:
            audio_path: 原始音频路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            speaker_id: 说话人ID
        
        Returns:
            提取的音频片段路径，失败返回None
        """
        try:
            # 创建临时文件
            temp_dir = Path(tempfile.gettempdir())
            output_path = temp_dir / f"speaker_{speaker_id}_{int(start_time)}.wav"
            
            # 使用ffmpeg提取片段
            cmd = [
                "ffmpeg",
                "-i", audio_path,
                "-ss", str(start_time),
                "-t", str(end_time - start_time),
                "-ac", "1",              # 单声道
                "-ar", "16000",          # 16kHz采样率
                "-y",
                "-loglevel", "error",
                str(output_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            return str(output_path)
            
        except FileNotFoundError:
            logger.error("❌ ffmpeg 未安装，无法提取音频片段")
            return None
        except subprocess.CalledProcessError as e:
            logger.error(f"❌ ffmpeg 提取失败: {e.stderr.decode() if e.stderr else str(e)}")
            return None
        except Exception as e:
            logger.error(f"❌ 提取音频片段异常: {e}")
            return None
    
    def match_speakers(self, 
                      speaker_segments: Dict[str, str],
                      threshold: float = 0.75) -> Dict[str, Tuple[str, str, float]]:
        """
        匹配说话人身份
        
        Args:
            speaker_segments: {speaker_id: audio_path}
            threshold: 相似度阈值（0-1）
        
        Returns:
            {speaker_id: (employee_id, name, similarity)}
        """
        if not self.enabled:
            return {}
        
        matched = {}
        
        for speaker_id, audio_path in speaker_segments.items():
            try:
                # 1. 提取声纹向量
                vector = self._extract_vector(audio_path)
                
                if vector is None:
                    logger.warning(f"⚠️ 说话人 {speaker_id} 声纹提取失败")
                    continue
                
                # 2. 在声纹库中搜索
                results = self.collection.query(
                    query_embeddings=[vector],
                    n_results=1
                )
                
                if not results['ids'] or len(results['ids'][0]) == 0:
                    logger.warning(f"⚠️ 说话人 {speaker_id} 未在声纹库中找到匹配")
                    continue
                
                # 3. 获取匹配结果
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
                try:
                    os.remove(audio_path)
                except:
                    pass
                
            except Exception as e:
                logger.error(f"❌ 匹配说话人 {speaker_id} 失败: {e}")
        
        return matched
    
    def _extract_vector(self, audio_path: str) -> Optional[List[float]]:
        """
        提取声纹向量
        
        Args:
            audio_path: 音频文件路径
        
        Returns:
            声纹向量（192维），失败返回None
        """
        try:
            res = self.embedding_model(audio_path)
            
            if res and 'spk_embedding' in res:
                vector = res['spk_embedding']
                
                # 转换为Python list
                if hasattr(vector, 'tolist'):
                    vector = vector.tolist()
                
                return vector
            else:
                logger.error(f"❌ 模型未返回 spk_embedding: {res}")
                return None
                
        except Exception as e:
            logger.error(f"❌ 提取声纹向量异常: {e}")
            return None
    
    def replace_speaker_ids(self,
                           transcript: List[Dict],
                           matched: Dict[str, Tuple[str, str, float]]) -> List[Dict]:
        """
        将speaker_id替换为真实姓名
        
        Args:
            transcript: ASR识别结果
            matched: 匹配结果 {speaker_id: (employee_id, name, similarity)}
        
        Returns:
            替换后的transcript
        """
        if not matched:
            return transcript
        
        for item in transcript:
            speaker_id = item.get("speaker_id", "unknown")
            
            if speaker_id in matched:
                employee_id, name, similarity = matched[speaker_id]
                item["speaker_name"] = name
                item["employee_id"] = employee_id
                item["voice_similarity"] = round(similarity, 3)
                logger.debug(f"替换: speaker_{speaker_id} → {name}")
        
        return transcript


# 全局单例
_voice_matcher = None


def get_voice_matcher() -> Optional[VoiceMatcher]:
    """获取声纹匹配器单例"""
    global _voice_matcher
    if _voice_matcher is None:
        try:
            _voice_matcher = VoiceMatcher()
        except Exception as e:
            logger.error(f"❌ 初始化声纹匹配器失败: {e}")
            return None
    return _voice_matcher
