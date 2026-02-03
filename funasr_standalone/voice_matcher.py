"""
声纹匹配服务
用于将ASR识别的speaker_id映射到真实员工姓名
"""
# 修复 datasets 兼容性问题（必须在导入 modelscope 之前）
def _fix_datasets_compatibility():
    """修复 datasets 与 modelscope 的兼容性问题"""
    try:
        import datasets
        
        # 修复 LargeList 导入
        if not hasattr(datasets, 'LargeList'):
            try:
                from datasets import LargeList
            except ImportError:
                try:
                    import pyarrow as pa
                    if hasattr(pa, 'large_list'):
                        datasets.LargeList = pa.large_list
                    elif hasattr(pa, 'LargeList'):
                        datasets.LargeList = pa.LargeList
                except Exception:
                    pass
        
        # 修复 _FEATURE_TYPES 导入（datasets 2.19+ 中可能已移除）
        try:
            from datasets.features.features import _FEATURE_TYPES
        except ImportError:
            try:
                # 尝试从新位置导入
                from datasets.features import _FEATURE_TYPES
            except ImportError:
                try:
                    # 如果不存在，创建一个兼容的占位符
                    import datasets.features.features as features_module
                    if not hasattr(features_module, '_FEATURE_TYPES'):
                        # 创建一个空的字典作为占位符
                        features_module._FEATURE_TYPES = {}
                except Exception:
                    pass
    except Exception:
        pass  # 如果 datasets 都导入不了，让后续代码自己处理错误

# 立即执行修复
_fix_datasets_compatibility()

import logging
import chromadb
import torch
from funasr import AutoModel
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
                 collection_name: str = "employee_voice_voiceprint",
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
            
            # 加载Cam++声纹模型（使用 FunASR AutoModel，直接输出 spk_embedding 向量）
            logger.info("📦 加载 Cam++ 声纹模型 (FunASR AutoModel)...")
            self.embedding_model = AutoModel(
                model="iic/speech_campplus_sv_zh-cn_16k-common",
                device=self.device,
                disable_update=True
            )
            logger.info("✅ 声纹模型加载成功（支持输出 spk_embedding）")
            
            # 连接ChromaDB
            logger.info(f"🔌 连接 ChromaDB: {chroma_host}:{chroma_port}")
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port
            )
            
            # 获取/创建专用的声纹库集合（Cam++，192维），与文本向量库完全隔离
            CAMPP_DIM = 192
            logger.info(f"🔌 连接声纹库集合: {collection_name} (期望维度: {CAMPP_DIM})")
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={
                    "hnsw:space": "cosine",
                    "embedding_dimension": CAMPP_DIM,
                    "model": "cam++"
                }
            )
            
            # 检查声纹库是否为空
            count = self.collection.count()
            if count == 0:
                logger.warning("⚠️ 声纹库为空，声纹识别将被禁用（请先录入员工声纹）")
                self.enabled = False
            else:
                logger.info(f"✅ 声纹库已就绪，共 {count} 个员工声纹（192维 Cam++）")
                self.enabled = True
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 声纹匹配器初始化失败: {error_msg}")
            
            # 根据错误类型给出具体的修复建议
            if "simplejson" in error_msg or "No module named 'simplejson'" in error_msg:
                logger.error("   💡 缺少依赖: simplejson")
                logger.error("   📦 解决方案: pip install simplejson")
            elif "sortedcontainers" in error_msg or "No module named 'sortedcontainers'" in error_msg:
                logger.error("   💡 缺少依赖: sortedcontainers")
                logger.error("   📦 解决方案: pip install sortedcontainers")
            elif "chromadb" in error_msg.lower() or "连接" in error_msg or "refused" in error_msg.lower():
                logger.error("   💡 ChromaDB 连接失败")
                logger.error("   📦 解决方案: 检查 ChromaDB 服务是否启动")
                logger.error("      启动命令: docker run -d --name chromadb -p 8000:8000 chromadb/chroma:latest")
            elif "datasets" in error_msg.lower() or "LargeList" in error_msg or "_FEATURE_TYPES" in error_msg:
                logger.error("   💡 datasets 版本兼容性问题")
                logger.error("   📦 解决方案: pip install 'datasets==2.17.0'")
            else:
                logger.error("   💡 请查看上方错误信息，根据错误类型修复")
            
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
            
            # 确保文件存在并返回绝对路径
            if output_path.exists():
                return str(output_path.resolve())
            else:
                logger.error(f"❌ 提取的音频文件不存在: {output_path}")
                return None
            
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
            # 确保路径是绝对路径字符串（Windows路径处理）
            original_path = audio_path
            logger.debug(f"🔍 _extract_vector 输入: {original_path}, 类型: {type(original_path)}")
            
            if isinstance(audio_path, Path):
                audio_path = str(audio_path.resolve())
            else:
                audio_path = str(Path(audio_path).resolve())
            
            logger.debug(f"🔍 处理后路径: {audio_path}")
            
            # 检查文件是否存在
            if not os.path.exists(audio_path):
                logger.error(f"❌ 音频文件不存在: {audio_path}")
                logger.error(f"   原始路径: {original_path}")
                logger.error(f"   路径类型: {type(original_path)}")
                return None
            
            logger.info(f"🔍 提取声纹向量: {audio_path}")
            logger.debug(f"   路径类型: {type(audio_path)}, 路径值: {repr(audio_path)}")
            
            try:
                # 使用 FunASR AutoModel 提取声纹向量
                # 与说话人分离模块保持一致：generate(input=audio_path)，返回列表，每个元素含 spk_embedding
                emb_res = self.embedding_model.generate(input=audio_path)
                logger.debug(f"   模型返回类型: {type(emb_res)}, 内容概要: {emb_res}")
                
                if not emb_res or len(emb_res) == 0:
                    logger.error("❌ 声纹模型未返回结果")
                    return None
                
                emb = emb_res[0].get("spk_embedding", None)
                if emb is None:
                    logger.error(f"❌ 模型未返回 spk_embedding，返回键: {list(emb_res[0].keys())}")
                    return None
                
                # 转为 numpy / list，并确保是一维向量
                import numpy as np
                emb_array = np.array(emb)
                if emb_array.ndim > 1:
                    emb_array = emb_array.flatten()
                
                vector = emb_array.tolist()
                logger.debug(f"   ✅ 成功提取声纹向量，维度: {len(vector)}")
                return vector
            
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ 声纹向量提取失败: {error_msg}")
                import traceback
                logger.debug(f"   详细错误: {traceback.format_exc()}")
                return None
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ 提取声纹向量异常: {error_msg}")
            logger.error(f"   音频路径: {audio_path if 'audio_path' in locals() else '未知'}")
            logger.error(f"   路径类型: {type(audio_path) if 'audio_path' in locals() else '未知'}")
            import traceback
            logger.debug(f"   详细错误: {traceback.format_exc()}")
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
