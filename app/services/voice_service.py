import logging
import chromadb
import torch
import tempfile
import traceback
import os
import uuid
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import time
from datetime import datetime
import numpy as np
import shutil

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

    def extract_vector(self, audio_path: str) -> Optional[List[float]]:
        """
        提取单个音频文件的声纹向量
        """
        import subprocess
        from pathlib import Path
        import soundfile as sf

        converted_audio_path: Optional[str] = None

        try:
            audio_path_obj = Path(audio_path)
            if not audio_path_obj.is_absolute():
                audio_path_obj = audio_path_obj.resolve()

            if not audio_path_obj.exists():
                logger.error(f"❌ 音频文件不存在: {audio_path_obj}")
                return None

            original_path = str(audio_path_obj)
            final_audio_path = original_path

            # 根据扩展名或 soundfile 能力判断是否需要转换
            file_ext = audio_path_obj.suffix.lower()
            unsupported_formats = {".m4a", ".aac", ".amr", ".opus", ".flac"}

            logger.info(f"📂 声纹提取 - 原始路径: {original_path}, 扩展名: {file_ext}")

            need_convert = False
            if file_ext in unsupported_formats:
                need_convert = True
            else:
                try:
                    sf.info(original_path)
                except Exception:
                    need_convert = True

            if need_convert:
                temp_dir = tempfile.gettempdir()
                temp_filename = f"voice_reg_{uuid.uuid4().hex}.wav"
                converted_audio_path = os.path.join(temp_dir, temp_filename)

                cmd = [
                    "ffmpeg",
                    "-i",
                    original_path,
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-f",
                    "wav",
                    "-y",
                    converted_audio_path,
                ]
                logger.info(f"🔄 声纹提取前使用 ffmpeg 转换为 WAV: {converted_audio_path}")
                try:
                    subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=60,
                    )
                except FileNotFoundError:
                    logger.error("❌ ffmpeg 未安装，无法进行音频格式转换")
                    return None
                except subprocess.CalledProcessError as ffmpeg_error:
                    error_msg = ffmpeg_error.stderr.decode() if ffmpeg_error.stderr else str(
                        ffmpeg_error
                    )
                    logger.error(f"❌ ffmpeg 转换失败: {error_msg}")
                    return None

                if not os.path.exists(converted_audio_path):
                    logger.error(f"❌ 转换后的文件不存在: {converted_audio_path}")
                    return None

                final_audio_path = converted_audio_path

            final_path_obj = Path(final_audio_path).resolve()
            final_audio_path_str = str(final_path_obj)

            if not final_path_obj.exists():
                logger.error(f"❌ 音频文件不存在: {final_audio_path_str}")
                return None

            logger.info(f"🔧 手动调用 modelscope 提取声纹, 路径: {final_audio_path_str}")

            try:
                # 1. 预处理：必须包在列表里传入
                pre_data = self.embedding_model.preprocess([final_audio_path_str])
                # 2. 前向传播：生成 embedding
                fwd_res = self.embedding_model.forward(pre_data)
                logger.info("✅ modelscope 前向传播成功")
            except Exception as e:
                logger.error(f"❌ modelscope 推理失败: {e}\n{traceback.format_exc()}")
                return None

            logger.debug(f"📊 fwd_res 类型: {type(fwd_res)}")

            vector = self._normalize_single_embedding(fwd_res)
            if vector is None:
                logger.error(
                    f"❌ 提取失败，无法解析底层返回结果，类型: {type(fwd_res)}"
                )
                return None

            logger.info(f"📐 提取向量成功，最终维度: {len(vector)}")
            return vector

        except Exception as e:
            logger.error(f"❌ 提取声纹向量异常: {e}\n{traceback.format_exc()}")
            return None
        finally:
            if converted_audio_path and os.path.exists(converted_audio_path):
                try:
                    os.remove(converted_audio_path)
                    logger.debug(
                        f"🧹 已清理格式转换临时文件: {converted_audio_path}"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ 清理格式转换临时文件失败: {e}")

    def _normalize_single_embedding(self, fwd_res):
        """
        将 modelscope forward 的输出统一规整为一条 1D 向量（list[float]）
        兼容多种返回结构：dict / Tensor / ndarray / list 等
        """
        vector = None
        
        # 情况A: 返回的是字典
        if isinstance(fwd_res, dict):
            candidate = None
            if 'embs' in fwd_res:
                candidate = fwd_res['embs']
            elif 'spk_embedding' in fwd_res:
                candidate = fwd_res['spk_embedding']
            elif 'embedding' in fwd_res:
                candidate = fwd_res['embedding']
            
            if candidate is not None:
                # 可能是单个向量、列表包向量，或 batch
                if hasattr(candidate, 'shape'):
                    # Tensor / ndarray
                    if len(candidate.shape) == 1:
                        vector = candidate
                    else:
                        # 取第一条
                        vector = candidate[0]
                elif isinstance(candidate, (list, tuple)):
                    # 如果是 list[vec]，取第一条；如果是单个向量 list[float]，直接用
                    if candidate and isinstance(candidate[0], (list, tuple)) or hasattr(candidate[0], 'shape'):
                        vector = candidate[0]
                    else:
                        vector = candidate
                else:
                    vector = candidate
        
        # 情况B: 直接返回 Tensor / ndarray
        if vector is None and hasattr(fwd_res, 'shape'):
            vector = fwd_res
        
        # 情况C: list / tuple
        if vector is None and isinstance(fwd_res, (list, tuple)) and len(fwd_res) > 0:
            vector = fwd_res[0]
        
        if vector is None:
            return None
        
        # 统一转为 1D list[float]
        if hasattr(vector, 'detach'):
            vector = vector.detach().cpu().numpy()
        elif hasattr(vector, 'numpy'):
            vector = vector.numpy()
        
        if hasattr(vector, 'tolist'):
            vector = vector.tolist()
        
        # 展平可能的 [[...]] 结构
        while isinstance(vector, list) and len(vector) == 1 and isinstance(vector[0], list):
            vector = vector[0]
        
        return vector
    
    
    def extract_vectors_batch(
        self, audio_paths: List[str], batch_size: int = 4
    ) -> List[Optional[List[float]]]:
        """
        小批量提取多个音频的声纹向量（batch_size <= 4）
        - 目标：在不改变功能的前提下，减少重复 IO 和模型初始化开销
        - 安全策略：如果 batch 处理失败，自动回落到逐条 extract_vector
        """
        if not audio_paths:
            return []
        
        # 强制限制 batch_size，防止显存/内存压力过大
        batch_size = max(1, min(4, int(batch_size)))
        
        results: List[Optional[List[float]]] = []
        
        try:
            for i in range(0, len(audio_paths), batch_size):
                batch = audio_paths[i:i + batch_size]
                norm_paths: List[str] = []
                for p in batch:
                    p_obj = Path(p)
                    if not p_obj.is_absolute():
                        p_obj = p_obj.resolve()
                    if not p_obj.exists():
                        logger.error(f"❌ 批量提取时音频文件不存在: {p_obj}")
                        results.append(None)
                        continue
                    norm_paths.append(str(p_obj))
                
                if not norm_paths:
                    continue
                
                logger.info(f"🔧 批量调用 modelscope 提取声纹，batch_size={len(norm_paths)}")
                try:
                    pre_data = self.embedding_model.preprocess(norm_paths)
                    fwd_res = self.embedding_model.forward(pre_data)
                except Exception as e:
                    logger.error(f"❌ 批量声纹提取失败，回落到逐条处理: {e}\n{traceback.format_exc()}")
                    # 回落为逐条调用，保证功能不受影响
                    for p in norm_paths:
                        results.append(self.extract_vector(p))
                    continue
                
                # 解析 batch 输出：转成若干条 1D 向量
                batch_vectors: List[Optional[List[float]]] = []
                if isinstance(fwd_res, dict):
                    candidate = None
                    if 'embs' in fwd_res:
                        candidate = fwd_res['embs']
                    elif 'spk_embedding' in fwd_res:
                        candidate = fwd_res['spk_embedding']
                    elif 'embedding' in fwd_res:
                        candidate = fwd_res['embedding']
                    
                    if candidate is not None:
                        # Tensor / ndarray
                        if hasattr(candidate, 'shape'):
                            try:
                                import numpy as np  # 局部引用，避免顶部循环依赖
                                arr = candidate
                                if len(arr.shape) == 1:
                                    batch_vectors.append(self._normalize_single_embedding({'embs': arr}))
                                else:
                                    for j in range(arr.shape[0]):
                                        batch_vectors.append(self._normalize_single_embedding({'embs': arr[j]}))
                            except Exception:
                                batch_vectors = [self._normalize_single_embedding(candidate)]
                        elif isinstance(candidate, (list, tuple)):
                            for item in candidate:
                                batch_vectors.append(self._normalize_single_embedding({'embs': item}))
                        else:
                            batch_vectors.append(self._normalize_single_embedding({'embs': candidate}))
                else:
                    # 非 dict：复用单条解析逻辑，尽量兼容
                    norm = self._normalize_single_embedding(fwd_res)
                    batch_vectors.append(norm)
                
                # 如果数量与输入不一致，做截断/填充，确保一一对应
                if len(batch_vectors) < len(norm_paths):
                    batch_vectors.extend([None] * (len(norm_paths) - len(batch_vectors)))
                elif len(batch_vectors) > len(norm_paths):
                    batch_vectors = batch_vectors[:len(norm_paths)]
                
                results.extend(batch_vectors)
        except Exception as e:
            logger.error(f"❌ 批量提取流程异常，整体回落到逐条处理: {e}\n{traceback.format_exc()}")
            results = [self.extract_vector(p) for p in audio_paths]
        
        return results
    
    def save_identity(self, employee_id: str, name: str, vector: list):
        """
        保存或更新员工声纹到 Chroma (支持声纹特征动量融合)
        """
        emp_id_str = str(employee_id)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        try:
            # 1. 尝试从 ChromaDB 中获取已有的声纹特征
            existing = self.collection.get(ids=[emp_id_str], include=["embeddings"])

            final_vector = vector
            update_type = "初次注册"

            if (
                existing
                and existing.get("embeddings")
                and len(existing["embeddings"]) > 0
            ):
                # 存在老特征，触发“特征融合”机制
                old_vector = existing["embeddings"][0]

                # 设置融合比例：保留 40% 的老特征，融入 60% 的新特征（会议实战环境音）
                alpha = 0.4
                old_np = np.array(old_vector)
                new_np = np.array(vector)

                # 向量线性相加
                fused_np = alpha * old_np + (1 - alpha) * new_np

                # L2 归一化，确保融合后的向量依然在单位球面上，不影响余弦相似度计算
                norm = np.linalg.norm(fused_np)
                if norm > 0:
                    fused_np = fused_np / norm

                final_vector = fused_np.tolist()
                update_type = "融合更新"
                logger.info(
                    f"🔄 检测到已存在 [{name}] 的声纹，正在进行特征融合 (新特征权重: 60%)..."
                )

            # 2. 使用 upsert 存入 ChromaDB（add 会报错，upsert 才能实现覆盖或新增）
            self.collection.upsert(
                ids=[emp_id_str],
                embeddings=[final_vector],
                metadatas=[
                    {
                        "name": name,
                        "employee_id": emp_id_str,
                        "update_time": current_time,
                        "update_type": update_type,
                    }
                ],
            )
            logger.info(
                f"💾 声纹已入库 ({update_type}): {name} (工号: {employee_id})"
            )
            return True

        except Exception as e:
            logger.error(f"❌ 声纹入库失败: {e}")
            raise e
    
    def delete_identity(self, employee_id: str) -> bool:
        """
        从 Chroma 声纹库中删除某个员工的声纹记录。
        
        Args:
            employee_id: 员工工号（作为 Chroma 的 id 使用）
        
        Returns:
            True 删除成功 / False 删除失败（具体错误可查看日志）
        """
        emp_id_str = str(employee_id)
        try:
            self.collection.delete(ids=[emp_id_str])
            logger.info(f"🗑️ 已从声纹库删除员工声纹: {emp_id_str}")
            return True
        except Exception as e:
            logger.error(f"❌ 删除声纹失败 (employee_id={emp_id_str}): {e}")
            return False
    
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
        import soundfile as sf

        if not self.enabled:
            return {}
        
        speaker_segments = {}
        speaker_times = {}  # {speaker_id: [(start, end), ...]}
        
        # 1. 收集每个说话人的所有时间段
        #    注意：外部 transcript 目前使用的是毫秒时间戳(start_time/end_time 为 ms)，
        #    这里需要统一转换为「秒」再参与后续采样点计算。
        for item in transcript:
            speaker_id = item.get("speaker_id", "unknown")
            start_raw = item.get("start_time", 0)
            end_raw = item.get("end_time", 0)

            # 兼容旧版（秒）与新版（毫秒）：
            # - 如果值大于 1000，认为是毫秒；否则认为是秒。
            try:
                start_val = float(start_raw)
                end_val = float(end_raw)
            except (TypeError, ValueError):
                start_val, end_val = 0.0, 0.0

            if start_val > 1000 or end_val > 1000:
                start_time = start_val / 1000.0
                end_time = end_val / 1000.0
            else:
                start_time = start_val
                end_time = end_val
            
            if speaker_id not in speaker_times:
                speaker_times[speaker_id] = []
            
            speaker_times[speaker_id].append((start_time, end_time))
        
        if not speaker_times:
            return {}
        
        # 2. 预处理整段音频：尽量只调用一次 ffmpeg / soundfile
        src_path = Path(audio_path).resolve()
        if not src_path.exists():
            logger.error(f"❌ 提取说话人片段时音频文件不存在: {src_path}")
            return {}
        
        temp_dir = Path(tempfile.gettempdir())
        converted_path = None
        working_path = src_path
        
        try:
            need_convert = False
            try:
                info = sf.info(str(src_path))
                # 不是 16k 单声道 WAV 时，统一转换一次
                if info.samplerate != 16000 or info.channels != 1 or info.format != "WAV":
                    need_convert = True
            except Exception:
                # soundfile 不支持该格式，使用 ffmpeg 转换一次
                need_convert = True
            
            if need_convert:
                converted_path = temp_dir / f"voice_seg_{uuid.uuid4().hex}.wav"
                cmd = [
                    "ffmpeg",
                    "-i", str(src_path),
                    "-ac", "1",
                    "-ar", "16000",
                    "-y",
                    "-loglevel", "error",
                    str(converted_path)
                ]
                logger.info(f"🔄 提取说话人片段前，统一转换音频为 16k 单声道 WAV: {converted_path}")
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                working_path = converted_path
            
            # 统一从 working_path 读取整段音频，只读一次
            audio_data, sr = sf.read(str(working_path))
            if audio_data is None or len(audio_data) == 0:
                logger.error("❌ 读取统一转换后的音频失败或为空")
                return {}

            # 如果是多通道，取第一通道
            if hasattr(audio_data, "ndim") and audio_data.ndim > 1:
                audio_data = audio_data[:, 0]

            logger.info(
                f"🔍 开始为 {len(speaker_times)} 个不同的speaker_id提取音频片段（每个ID只提取一次，统一读盘）"
            )

            for speaker_id, times in speaker_times.items():
                if speaker_id == "unknown":
                    continue

                try:
                    # 找出该说话人最长的连续片段
                    sorted_times = sorted(
                        times, key=lambda x: x[1] - x[0], reverse=True
                    )

                    # 取第一段（最长的）- 每个speaker_id只提取一次
                    if sorted_times:
                        start, end = sorted_times[0]
                        segment_end = min(end, start + duration)

                        start_idx = max(0, int(start * sr))
                        end_idx = max(start_idx + 1, int(segment_end * sr))
                        end_idx = min(end_idx, len(audio_data))

                        if end_idx <= start_idx:
                            continue

                        segment = audio_data[start_idx:end_idx]

                        output_path = temp_dir / f"speaker_{speaker_id}_{int(start)}.wav"
                        sf.write(str(output_path), segment, sr)

                        if output_path.exists():
                            speaker_segments[speaker_id] = str(
                                output_path.resolve()
                            )
                            logger.info(
                                f"✅ 提取说话人 {speaker_id} 音频片段: {start:.1f}s - {segment_end:.1f}s"
                            )

                except Exception as e:
                    logger.error(f"❌ 提取说话人 {speaker_id} 音频失败: {e}")

        except Exception as e:
            logger.error(f"❌ 统一提取说话人片段流程异常，将回退到逐 speaker ffmpeg 方案: {e}")
            speaker_segments = {}
            # 回退逻辑：保持原先每个 speaker 独立调用 ffmpeg 的实现，保证功能不变
            for speaker_id, times in speaker_times.items():
                if speaker_id == "unknown":
                    continue
                try:
                    sorted_times = sorted(
                        times, key=lambda x: x[1] - x[0], reverse=True
                    )
                    if sorted_times:
                        start, end = sorted_times[0]
                        segment_end = min(end, start + duration)

                        temp_dir_fallback = Path(tempfile.gettempdir())
                        output_path = (
                            temp_dir_fallback
                            / f"speaker_{speaker_id}_{int(start)}.wav"
                        )

                        cmd = [
                            "ffmpeg",
                            "-i",
                            audio_path,
                            "-ss",
                            str(start),
                            "-t",
                            str(segment_end - start),
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-y",
                            "-loglevel",
                            "error",
                            str(output_path),
                        ]
                        subprocess.run(
                            cmd, check=True, capture_output=True, timeout=30
                        )
                        if output_path.exists():
                            speaker_segments[speaker_id] = str(
                                output_path.resolve()
                            )
                            logger.info(
                                f"✅ [回退方案] 提取说话人 {speaker_id} 音频片段: {start:.1f}s - {segment_end:.1f}s"
                            )
                except Exception as e2:
                    logger.error(
                        f"❌ [回退方案] 提取说话人 {speaker_id} 音频失败: {e2}"
                    )
        finally:
            # 清理统一转换产生的临时文件
            if converted_path and converted_path.exists():
                try:
                    converted_path.unlink()
                    logger.debug(f"🧹 已清理统一转换临时文件: {converted_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理统一转换临时文件失败: {e}")
        
        return speaker_segments
    
    def match_speakers(self, speaker_segments: Dict[str, str], threshold: float = 0.60) -> Dict[str, tuple]:
        """
        匹配说话人身份：引入全局排序与排他机制，防止 Pyannote 区分出的人被合并
        """
        if not self.enabled:
            return {}
        
        logger.info(f"🔍 开始匹配 {len(speaker_segments)} 个不同的speaker_id（全局排他模式）")
        
        # 1. 收集所有候选人的打分情况
        candidates = []
        for speaker_id, audio_data in speaker_segments.items():
            try:
                audio_paths = [audio_data] if isinstance(audio_data, str) else audio_data
                if not isinstance(audio_paths, list):
                    continue
                
                # 使用小批量接口提取向量，避免重复 IO 和重复模型调用
                vectors_all = self.extract_vectors_batch(audio_paths, batch_size=4)
                vectors = [v for v in vectors_all if v is not None]
                if not vectors:
                    candidates.append({"speaker_id": speaker_id, "sim": 0.0, "emp_id": None, "name": None})
                    continue
                
                # 计算均值向量
                import numpy as np
                mean_vector = np.mean(np.array(vectors), axis=0).tolist() if len(vectors) > 1 else vectors[0]
                
                # 在库中搜索最接近的 1 个人
                results = self.collection.query(query_embeddings=[mean_vector], n_results=1)
                
                if not results['ids'] or len(results['ids'][0]) == 0:
                    candidates.append({"speaker_id": speaker_id, "sim": 0.0, "emp_id": None, "name": None})
                    continue
                
                emp_id = results['ids'][0][0]
                name = results['metadatas'][0][0].get('name', '未知')
                distance = results['distances'][0][0] if 'distances' in results else 0.5
                sim = 1 - (distance / 2.0)
                
                candidates.append({
                    "speaker_id": speaker_id, "sim": sim, "emp_id": emp_id, "name": name,
                    "audio_paths": audio_paths # 👈 【修改1】把音频路径存进字典里传给下一步
                })
                
                # 清理临时文件
                # 清理临时文件
                # for p in audio_paths:
                #     import os, tempfile
                #     if os.path.exists(p) and p.startswith(tempfile.gettempdir()):
                #         try:
                #             os.remove(p)
                #         except:
                #             pass
                        
            except Exception as e:
                logger.error(f"❌ 预处理说话人 {speaker_id} 失败: {e}")
                candidates.append({"speaker_id": speaker_id, "sim": 0.0, "emp_id": None, "name": None})

        # 2. 按照相似度从高到低排序 (至关重要！让得分高的人先挑选身份)
        candidates.sort(key=lambda x: x["sim"], reverse=True)
        
        matched = {}
        used_employee_ids = set() # 用于记录已经被占用的身份

        unclaimed_dir = os.path.join(os.getcwd(), "unclaimed_voices")
        os.makedirs(unclaimed_dir, exist_ok=True)
        
        # 3. 依次分配身份
        # 3. 依次分配身份
        for cand in candidates:
            speaker_id = cand["speaker_id"]
            sim = cand["sim"]
            emp_id = cand["emp_id"]
            name = cand["name"]
            audio_paths = cand.get("audio_paths", []) # 👈 获取刚才存下来的路径

            if (
                sim >= threshold
                and emp_id is not None
                and emp_id not in used_employee_ids
            ):
                matched[speaker_id] = (emp_id, name, sim)
                used_employee_ids.add(emp_id)
                logger.info(f"✅ 说话人 {speaker_id} 锁定身份: {name} (相似度: {sim:.2%})")
                
                # 匹配成功，音频没用了，正常清理
                for p in audio_paths:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
            else:
                # 没达到及格线，兜底为未知发言人
                reason = "相似度不足" if sim < threshold else f"身份 [{name}] 已被占用"
                logger.warning(f"⚠️ 说话人 {speaker_id} 降级为未知 ({reason}, sim: {sim:.2%})")
                matched[speaker_id] = (f"unknown_{speaker_id}", f"未知发言人_{speaker_id}", sim)
                
                # 👇 【核心新增】如果是未知发言人，把它的第一段有效音频拷贝留下来
                if audio_paths and os.path.exists(audio_paths[0]):
                    # 文件名格式: unknown_SPEAKER_01_sim59.wav
                    save_path = os.path.join(unclaimed_dir, f"unknown_{speaker_id}_sim{int(sim*100)}.wav")
                    try:
                        shutil.copy2(audio_paths[0], save_path)
                        logger.info(f"💾 已截获待认领音频: {save_path}")
                    except Exception as e:
                        logger.error(f"保存待认领音频失败: {e}")
                
                # 拷贝完成后，清理原本的临时文件防内存泄漏
                for p in audio_paths:
                    if os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
        
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
                # 不添加 employee_id 和 similarity 到 transcript_item 中
        
        return transcript


try:
    voice_service = VoiceService()
except Exception as e:
    logger.error(f"⚠️ VoiceService 初始化失败: {e}")
    voice_service = None

def get_voice_service():
    if voice_service is None:
        raise RuntimeError("VoiceService 未成功初始化，请检查日志")
    return voice_service