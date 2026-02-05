#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyannote 说话人分离模块
使用专业的 Pyannote.audio 模型进行说话人分离
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path
import os
import shutil
import tempfile

import torch
import soundfile as sf
import subprocess
import tempfile

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("⚠️ Pyannote.audio 未安装，说话人分离功能将不可用")
    logger.warning("   安装命令: pip install pyannote.audio")


# 全局 pipeline 缓存（避免重复加载）
_pipeline_cache = None


def _extract_annotation(diarization):
    """
    兼容不同版本 Pyannote：
    - 旧版: pipeline(...) 直接返回支持 itertracks 的 Annotation
    - 3.x: 可能返回 DiarizeOutput，真正的 Annotation 在某个属性里
    """
    # 旧版：直接就是 Annotation
    if hasattr(diarization, "itertracks"):
        return diarization

    # 尝试通过 __dict__ 或 vars() 访问（适用于 dataclass/NamedTuple）
    try:
        obj_dict = vars(diarization) if hasattr(diarization, "__dict__") else {}
        for key, value in obj_dict.items():
            if value is not None and hasattr(value, "itertracks"):
                return value
    except:
        pass

    # 尝试通过索引访问（如果是 NamedTuple）
    try:
        if hasattr(diarization, "__len__"):
            for i in range(len(diarization)):
                ann = diarization[i]
                if ann is not None and hasattr(ann, "itertracks"):
                    return ann
    except:
        pass

    # 新版：DiarizeOutput dataclass - 尝试多种可能的属性名
    possible_attrs = ["annotation", "speaker", "output", "result", "diarization", "labels"]
    ann = None
    
    for attr in possible_attrs:
        try:
            ann = getattr(diarization, attr, None)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
        except:
            continue
    
    # 如果是 dataclass，尝试访问所有字段
    if hasattr(diarization, "__dataclass_fields__"):
        for field_name in diarization.__dataclass_fields__.keys():
            try:
                ann = getattr(diarization, field_name, None)
                if ann is not None and hasattr(ann, "itertracks"):
                    return ann
            except:
                continue
    
    # 尝试 dir() 查找所有属性
    for attr_name in dir(diarization):
        if attr_name.startswith("_"):
            continue
        try:
            ann = getattr(diarization, attr_name)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann
        except:
            continue

    # 有些版本可能返回 dict
    if isinstance(diarization, dict):
        for key in ["annotation", "speaker", "output", "result", "diarization"]:
            ann = diarization.get(key)
            if ann is not None and hasattr(ann, "itertracks"):
                return ann

    # 如果还是找不到，打印所有属性以便调试
    attrs = [a for a in dir(diarization) if not a.startswith("_")]
    try:
        obj_dict = vars(diarization) if hasattr(diarization, "__dict__") else {}
        logger.error(f"DiarizeOutput 对象详情: {obj_dict}")
    except:
        pass
    
    raise TypeError(
        f"Unsupported diarization output type: {type(diarization)}\n"
        f"Available attributes: {attrs}\n"
        f"Type: {type(diarization).__name__}\n"
        f"请检查 Pyannote 版本和 DiarizeOutput 的实际结构"
    )


def detect_long_silence_with_vad(waveform: torch.Tensor, sample_rate: int, min_silence_duration: float = 2.0) -> List[tuple]:
    """
    使用轻量级 VAD 检测长静音点（>2秒）- 优化版本
    
    Args:
        waveform: 音频波形 tensor，形状为 [channels, time]
        sample_rate: 采样率
        min_silence_duration: 最小静音时长（秒），默认 2.0 秒
    
    Returns:
        静音段列表 [(start_time, end_time), ...]，单位：秒
    """
    logger.info(f"🔍 检测长静音点（>={min_silence_duration}秒）...")
    
    try:
        # 优化：下采样以加快处理速度并降低内存占用
        # 对于静音检测，不需要高精度，可以大幅降低采样率
        downsample_factor = 8  # 降低到 1/8 采样率（进一步降低内存占用）
        if downsample_factor > 1:
            # 简单下采样：每隔 N 个点取一个
            if waveform.ndim > 1:
                downsampled = waveform[:, ::downsample_factor]
            else:
                downsampled = waveform[::downsample_factor]
            effective_sample_rate = sample_rate / downsample_factor
        else:
            downsampled = waveform
            effective_sample_rate = sample_rate
        
        # 计算音频能量（RMS）- 使用批量计算
        if downsampled.ndim > 1:
            audio_energy = torch.sqrt(torch.mean(downsampled ** 2, dim=0))
        else:
            audio_energy = torch.abs(downsampled)
        
        # 归一化
        max_energy = torch.max(audio_energy)
        if max_energy > 0:
            audio_energy = audio_energy / max_energy
        
        # 使用更大的帧窗口以加快处理并降低内存占用（2秒一帧）
        frame_duration = 2.0  # 改为 2 秒一帧，进一步降低内存占用
        frame_samples = int(effective_sample_rate * frame_duration)
        
        # 批量计算每帧的平均能量（使用更小的批次）
        num_frames = len(audio_energy) // frame_samples
        if num_frames > 0:
            # 分批处理，避免一次性加载所有帧到内存
            batch_size = 1000  # 每批处理 1000 帧
            frame_energies_list = []
            for i in range(0, num_frames, batch_size):
                end_idx = min(i + batch_size, num_frames)
                batch_frames = audio_energy[i * frame_samples:end_idx * frame_samples]
                if len(batch_frames) >= frame_samples:
                    batch_frames = batch_frames[:len(batch_frames) // frame_samples * frame_samples]
                    frames = batch_frames.view(-1, frame_samples)
                    frame_energies_list.append(torch.mean(frames, dim=1))
            if frame_energies_list:
                frame_energies = torch.cat(frame_energies_list)
            else:
                frame_energies = torch.tensor([torch.mean(audio_energy)])
        else:
            frame_energies = torch.tensor([torch.mean(audio_energy)])
        
        # 检测静音（能量低于阈值）- 提高阈值，减少误检
        energy_threshold = 0.05  # 提高阈值到 0.05
        
        # 找到连续静音段（使用批量操作）
        silence_frames = frame_energies < energy_threshold
        
        # 使用 torch 的批量操作找到静音段
        silence_segments = []
        in_silence = False
        silence_start = 0
        
        for i in range(len(silence_frames)):
            if silence_frames[i] and not in_silence:
                in_silence = True
                silence_start = i * frame_duration
            elif not silence_frames[i] and in_silence:
                in_silence = False
                silence_end = i * frame_duration
                if silence_end - silence_start >= min_silence_duration:
                    silence_segments.append((silence_start, silence_end))
        
        # 处理最后一个静音段
        if in_silence:
            silence_end = len(silence_frames) * frame_duration
            if silence_end - silence_start >= min_silence_duration:
                silence_segments.append((silence_start, silence_end))
        
        logger.info(f"✅ 检测到 {len(silence_segments)} 个长静音段（>={min_silence_duration}秒）")
        return silence_segments
        
    except Exception as e:
        logger.warning(f"⚠️ VAD 检测失败: {e}")
        return []


def split_audio_by_silence(waveform: torch.Tensor, sample_rate: int, 
                           silence_segments: List[tuple],
                           min_chunk_duration: float = 600.0,  # 10 分钟
                           max_chunk_duration: float = 1200.0) -> List[tuple]:
    """
    根据静音点将音频切分为 10-20 分钟的片段（优化版）
    
    Args:
        waveform: 音频波形 tensor
        sample_rate: 采样率
        silence_segments: 静音段列表 [(start, end), ...]
        min_chunk_duration: 最小片段时长（秒），默认 600 秒（10 分钟）
        max_chunk_duration: 最大片段时长（秒），默认 1200 秒（20 分钟）
    
    Returns:
        音频片段列表 [(start_time, end_time), ...]，单位：秒
    """
    audio_duration = waveform.shape[-1] / sample_rate
    chunks = []
    
    if not silence_segments or len(silence_segments) > 1000:
        # 静音段太多或没有，直接按最大时长切分（避免过度切分）
        logger.info(f"⚠️ 静音段数量异常（{len(silence_segments) if silence_segments else 0}），使用固定时长切分")
        num_chunks = int(audio_duration / max_chunk_duration) + 1
        for i in range(num_chunks):
            start = i * max_chunk_duration
            end = min((i + 1) * max_chunk_duration, audio_duration)
            chunks.append((start, end))
        logger.info(f"✂️ 切分为 {len(chunks)} 个片段（每段 {max_chunk_duration}秒）")
        return chunks
    
    # 优化：只选择足够长的静音段（>3秒）作为切分点
    valid_silence_points = []
    for silence_start, silence_end in silence_segments:
        silence_duration = silence_end - silence_start
        if silence_duration >= 3.0:  # 只使用 >3秒 的静音段
            # 使用静音段的中间点作为切分点
            cut_point = (silence_start + silence_end) / 2
            valid_silence_points.append(cut_point)
    
    # 根据有效静音点切分
    current_start = 0.0
    
    for cut_point in valid_silence_points:
        chunk_duration = cut_point - current_start
        
        # 如果达到最小时长，在切分点切分
        if chunk_duration >= min_chunk_duration:
            chunks.append((current_start, cut_point))
            current_start = cut_point
        
        # 如果超过最大时长，强制切分
        if chunk_duration >= max_chunk_duration:
            chunks.append((current_start, cut_point))
            current_start = cut_point
    
    # 添加最后一个片段
    if audio_duration - current_start > 0:
        chunks.append((current_start, audio_duration))
    
    # 如果切分结果太少，回退到固定切分
    if len(chunks) == 0 or (len(chunks) == 1 and chunks[0][1] - chunks[0][0] > max_chunk_duration * 2):
        logger.info("⚠️ 切分结果不理想，回退到固定时长切分")
        chunks = []
        num_chunks = int(audio_duration / max_chunk_duration) + 1
        for i in range(num_chunks):
            start = i * max_chunk_duration
            end = min((i + 1) * max_chunk_duration, audio_duration)
            chunks.append((start, end))
    
    logger.info(f"✂️ 根据静音点切分为 {len(chunks)} 个片段（10-20分钟）")
    return chunks


def extract_speaker_embeddings(pipeline, waveform: torch.Tensor, sample_rate: int, 
                               annotation) -> Dict[str, torch.Tensor]:
    """
    从 diarization 结果中提取每个 speaker 的 embedding
    
    Args:
        pipeline: Pyannote pipeline 对象
        waveform: 音频波形 tensor
        sample_rate: 采样率
        annotation: Pyannote Annotation 对象
    
    Returns:
        {speaker_id: embedding_tensor, ...}
    """
    speaker_embeddings = {}
    
    try:
        # 获取 embedding 模型
        # Pyannote pipeline 的 embedding 可能通过多种方式访问
        embedding_model = None
        
        # 方法1：直接访问 pipeline.embedding
        if hasattr(pipeline, "embedding"):
            embedding_attr = pipeline.embedding
            if isinstance(embedding_attr, str):
                try:
                    from pyannote.audio import Model
                    embedding_model = Model.from_pretrained(embedding_attr)
                    if torch.cuda.is_available():
                        embedding_model = embedding_model.to(torch.device("cuda"))
                    logger.info(f"✅ 从路径加载 embedding 模型: {embedding_attr}")
                except Exception as e:
                    logger.debug(f"⚠️ 无法从路径加载 embedding 模型: {e}")
            elif hasattr(embedding_attr, "__call__") or hasattr(embedding_attr, "forward"):
                embedding_model = embedding_attr
                logger.info("✅ 使用 pipeline.embedding 模型对象")
        
        # 方法2：访问 pipeline._embedding
        if embedding_model is None and hasattr(pipeline, "_embedding"):
            embedding_attr = pipeline._embedding
            if isinstance(embedding_attr, str):
                try:
                    from pyannote.audio import Model
                    embedding_model = Model.from_pretrained(embedding_attr)
                    if torch.cuda.is_available():
                        embedding_model = embedding_model.to(torch.device("cuda"))
                    logger.info(f"✅ 从路径加载 embedding 模型: {embedding_attr}")
                except Exception as e:
                    logger.debug(f"⚠️ 无法从路径加载 embedding 模型: {e}")
            elif hasattr(embedding_attr, "__call__") or hasattr(embedding_attr, "forward"):
                embedding_model = embedding_attr
                logger.info("✅ 使用 pipeline._embedding 模型对象")
        
        # 方法3：通过 pipeline 的内部结构访问（尝试多种可能的属性名）
        if embedding_model is None:
            possible_attrs = ["embedding_model", "_embedding_model", "speaker_embedding", "_speaker_embedding"]
            for attr_name in possible_attrs:
                if hasattr(pipeline, attr_name):
                    attr_value = getattr(pipeline, attr_name)
                    if hasattr(attr_value, "__call__") or hasattr(attr_value, "forward"):
                        embedding_model = attr_value
                        logger.info(f"✅ 通过 {attr_name} 获取 embedding 模型")
                        break
        
        # 方法4：从 pipeline 的 embedding 组件获取（如果 pipeline 有 embedding 组件）
        if embedding_model is None and hasattr(pipeline, "components"):
            components = pipeline.components
            if isinstance(components, dict) and "embedding" in components:
                embedding_comp = components["embedding"]
                if hasattr(embedding_comp, "model"):
                    embedding_model = embedding_comp.model
                    logger.info("✅ 通过 pipeline.components['embedding'].model 获取 embedding 模型")
                elif hasattr(embedding_comp, "__call__") or hasattr(embedding_comp, "forward"):
                    embedding_model = embedding_comp
                    logger.info("✅ 通过 pipeline.components['embedding'] 获取 embedding 模型")
        
        if embedding_model is None:
            logger.warning("⚠️ 无法获取 embedding 模型，跳过声纹提取（不影响主流程）")
            logger.debug(f"⚠️ Pipeline 属性: {[attr for attr in dir(pipeline) if 'embed' in attr.lower()]}")
            return speaker_embeddings
        
        # 对每个 speaker segment 提取 embedding
        from pyannote.core import Segment
        
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            # 确保 speaker 不为 None
            if speaker is None or speaker == "":
                speaker = "SPEAKER_UNKNOWN"
            
            # 提取该 segment 的音频
            start_sample = int(turn.start * sample_rate)
            end_sample = int(turn.end * sample_rate)
            segment_waveform = waveform[:, start_sample:end_sample]
            
            # 确保 segment_waveform 在正确的设备上
            if hasattr(embedding_model, "device"):
                model_device = embedding_model.device
            elif hasattr(embedding_model, "_device"):
                model_device = embedding_model._device
            else:
                model_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            if segment_waveform.device != model_device:
                segment_waveform = segment_waveform.to(model_device)
            
            # 提取 embedding
            with torch.no_grad():
                try:
                    # 尝试不同的调用方式
                    embedding = None
                    
                    # 方式1：直接调用（__call__）
                    if hasattr(embedding_model, "__call__"):
                        try:
                            embedding = embedding_model({"waveform": segment_waveform, "sample_rate": sample_rate})
                        except Exception as e1:
                            logger.debug(f"⚠️ __call__ 方式失败: {e1}")
                    
                    # 方式2：forward 方法
                    if embedding is None and hasattr(embedding_model, "forward"):
                        try:
                            embedding = embedding_model.forward({"waveform": segment_waveform, "sample_rate": sample_rate})
                        except Exception as e2:
                            logger.debug(f"⚠️ forward 方式失败: {e2}")
                    
                    # 方式3：使用 pipeline 的 embedding 方法（如果有）
                    if embedding is None and hasattr(pipeline, "embedding") and callable(pipeline.embedding):
                        try:
                            embedding = pipeline.embedding({"waveform": segment_waveform, "sample_rate": sample_rate})
                        except Exception as e3:
                            logger.debug(f"⚠️ pipeline.embedding 方式失败: {e3}")
                    
                    if embedding is None:
                        continue
                    
                    # embedding 可能是 tensor 或 dict，需要处理
                    if isinstance(embedding, dict):
                        embedding = embedding.get("embedding", embedding.get("output", embedding.get("logits", embedding.get("embeddings", None))))
                    if embedding is None or not isinstance(embedding, torch.Tensor):
                        continue
                except Exception as e:
                    logger.debug(f"⚠️ 提取 segment embedding 失败: {e}")
                    continue
                
                # 如果是多帧，取平均
                if embedding.ndim > 1:
                    embedding = torch.mean(embedding, dim=0)
                
                # 累积该 speaker 的 embedding（取平均）
                if speaker not in speaker_embeddings:
                    speaker_embeddings[speaker] = []
                speaker_embeddings[speaker].append(embedding)
        
        # 对每个 speaker 的多个 embedding 取平均
        for speaker in speaker_embeddings:
            embeddings_list = speaker_embeddings[speaker]
            if len(embeddings_list) > 1:
                speaker_embeddings[speaker] = torch.mean(torch.stack(embeddings_list), dim=0)
            else:
                speaker_embeddings[speaker] = embeddings_list[0]
        
        logger.info(f"✅ 提取了 {len(speaker_embeddings)} 个 speaker 的 embedding")
        
    except Exception as e:
        logger.warning(f"⚠️ 提取 speaker embedding 失败: {e}")
    
    return speaker_embeddings


def global_speaker_calibration(all_chunk_results: List[Dict], 
                               global_embeddings: Dict[str, torch.Tensor],
                               threshold: float = 0.7) -> Dict[str, str]:
    """
    全局声纹校准：对跨片段的 speaker ID 进行聚类校准
    
    Args:
        all_chunk_results: 所有片段的 diarization 结果
            [{"chunk_idx": 0, "annotation": Annotation, "embeddings": {...}}, ...]
        global_embeddings: 全局 speaker embedding 字典
            {"chunk_0_SPEAKER_00": embedding, "chunk_1_SPEAKER_01": embedding, ...}
        threshold: 聚类阈值（余弦相似度）
    
    Returns:
        speaker ID 映射字典 {"chunk_0_SPEAKER_00": "SPEAKER_00", ...}
    """
    try:
        try:
            from sklearn.cluster import AgglomerativeClustering
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            logger.warning("⚠️ sklearn 未安装，无法进行全局校准，请安装: pip install scikit-learn")
            return {}
        import numpy as np
        
        if not global_embeddings:
            logger.warning("⚠️ 没有 embedding，跳过全局校准")
            return {}
        
        # 准备数据
        speaker_ids = list(global_embeddings.keys())
        embeddings_matrix = torch.stack([global_embeddings[sid] for sid in speaker_ids]).cpu().numpy()
        
        # 计算相似度矩阵
        similarity_matrix = cosine_similarity(embeddings_matrix)
        
        # 使用层次聚类
        # 距离 = 1 - 相似度
        distance_matrix = 1 - similarity_matrix
        
        # 聚类
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=1 - threshold,  # 距离阈值
            linkage='average',
            metric='precomputed'
        )
        cluster_labels = clustering.fit_predict(distance_matrix)
        
        # 创建映射：每个 speaker_id -> 全局统一的 speaker_id
        # 使用每个 cluster 中第一个 speaker 的 ID 作为代表
        id_mapping = {}
        cluster_to_global_id = {}
        global_id_counter = 0
        
        for i, speaker_id in enumerate(speaker_ids):
            cluster_id = cluster_labels[i]
            
            if cluster_id not in cluster_to_global_id:
                # 创建新的全局 speaker ID
                global_speaker_id = f"SPEAKER_{global_id_counter:02d}"
                cluster_to_global_id[cluster_id] = global_speaker_id
                global_id_counter += 1
            
            id_mapping[speaker_id] = cluster_to_global_id[cluster_id]
        
        logger.info(f"✅ 全局校准完成：{len(speaker_ids)} 个片段 speaker -> {global_id_counter} 个全局 speaker")
        
        return id_mapping
        
    except Exception as e:
        logger.warning(f"⚠️ 全局校准失败: {e}")
        return {}


def process_audio_with_pipeline(pipeline, waveform: torch.Tensor, sample_rate: int, 
                                max_chunk_duration: int = 300,
                                use_vad_smart_chunking: bool = True):
    """
    使用 Pyannote pipeline 处理音频（支持长音频分段处理）
    
    Args:
        pipeline: Pyannote pipeline 对象（应该已经在正确的设备上）
        waveform: 音频波形 tensor，形状为 [channels, time]（应该已经在正确的设备上）
        sample_rate: 采样率
        max_chunk_duration: 最大片段时长（秒），超过此长度会分段处理
    
    Returns:
        diarization 结果（Annotation 或 DiarizeOutput）
    """
    audio_duration = waveform.shape[-1] / sample_rate
    
    # 检查并确认 GPU 使用情况（从 pipeline 或 waveform 获取设备）
    if hasattr(pipeline, "device"):
        device = pipeline.device
    elif hasattr(pipeline, "_device"):
        device = pipeline._device
    else:
        # 从 waveform 推断设备，或使用默认
        device = waveform.device if hasattr(waveform, "device") else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    device_info = "GPU" if device.type == "cuda" else "CPU"
    
    if device.type == "cuda" and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_allocated = torch.cuda.memory_allocated(0) / 1024**3
        gpu_memory_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"🖥️ 使用设备: {device_info} ({gpu_name})")
        logger.info(f"💾 GPU 显存: 已分配 {gpu_memory_allocated:.2f}GB / 已保留 {gpu_memory_reserved:.2f}GB")
    else:
        logger.info(f"🖥️ 使用设备: {device_info} (CUDA 不可用或未使用)")
    
    # 确保 waveform 在正确的设备上
    if waveform.device != device:
        logger.info(f"🔄 将 waveform 从 {waveform.device} 移动到 {device}")
        waveform = waveform.to(device)
    
    # 短音频直接处理（使用优化后的 pipeline）
    if audio_duration <= max_chunk_duration:
        logger.info(f"⏱️ 音频时长 {audio_duration:.1f}秒，直接处理")
        return pipeline({"waveform": waveform, "sample_rate": sample_rate})
    
    # 长音频：优化策略
    # 对于超长音频（>1小时），跳过 VAD 切分，直接固定切分 + 全局校准
    # VAD 切分耗时且可能产生过多片段，固定切分更高效
    vad_chunking_success = False
    chunk_boundaries = None
    
    if use_vad_smart_chunking and audio_duration < 3600:
        # 只对 <1小时 的音频使用 VAD 切分
        logger.info(f"⏱️ 音频时长 {audio_duration:.1f}秒，使用 VAD 智能分段 + 全局声纹校准")
        try:
            from pyannote.core import Annotation, Segment
            
            # 第一步：VAD 检测长静音点（>2秒）
            silence_segments = detect_long_silence_with_vad(waveform, sample_rate, min_silence_duration=2.0)
            
            # 第二步：根据静音点切分为 10-20 分钟片段
            chunk_boundaries = split_audio_by_silence(
                waveform, sample_rate, silence_segments,
                min_chunk_duration=600.0,  # 10 分钟
                max_chunk_duration=1200.0   # 20 分钟
            )
            
            # 如果切分结果不理想（片段太多或太少），回退到固定切分
            if not chunk_boundaries or len(chunk_boundaries) == 1 or len(chunk_boundaries) > 50:
                logger.info(f"⚠️ VAD 切分结果不理想（{len(chunk_boundaries) if chunk_boundaries else 0}个片段），回退到固定切分")
                use_vad_smart_chunking = False
                chunk_boundaries = None
            else:
                logger.info(f"✅ VAD 切分成功，共 {len(chunk_boundaries)} 个片段")
                vad_chunking_success = True
        except Exception as e:
            logger.warning(f"⚠️ VAD 切分失败: {e}，回退到固定切分")
            use_vad_smart_chunking = False
            chunk_boundaries = None
    
    # 如果 VAD 切分成功，使用 VAD 切分的结果进行处理
    if vad_chunking_success and chunk_boundaries:
        # 使用 VAD 切分的结果进行处理
        try:
            from pyannote.core import Annotation, Segment
            
            # 第三步：分段推理（优化内存使用）
            all_chunk_results = []
            global_embeddings = {}
            
            logger.info(f"🚀 开始处理 {len(chunk_boundaries)} 个片段（VAD 切分）...")
            
            for chunk_idx, (chunk_start_time, chunk_end_time) in enumerate(chunk_boundaries):
                # 计算采样点
                start_sample = int(chunk_start_time * sample_rate)
                end_sample = int(chunk_end_time * sample_rate)
                chunk_waveform = waveform[:, start_sample:end_sample].clone()
                
                if chunk_waveform.shape[-1] < sample_rate * 0.5:
                    logger.warning(f"⚠️ 片段 {chunk_idx + 1} 太短（{chunk_waveform.shape[-1] / sample_rate:.2f}秒），跳过")
                    continue
                
                if chunk_waveform.device != device:
                    chunk_waveform = chunk_waveform.to(device)
                
                if chunk_idx % 5 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
                
                chunk_duration = chunk_end_time - chunk_start_time
                logger.info(f"🔄 处理片段 {chunk_idx + 1}/{len(chunk_boundaries)} ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s, 时长: {chunk_duration:.1f}s) [设备: {chunk_waveform.device}]")
                
                chunk_diarization = pipeline({"waveform": chunk_waveform, "sample_rate": sample_rate})
                chunk_annotation = _extract_annotation(chunk_diarization)
                
                chunk_embeddings = {}
                try:
                    chunk_embeddings = extract_speaker_embeddings(
                        pipeline, chunk_waveform, sample_rate, chunk_annotation
                    )
                except Exception as e:
                    logger.debug(f"⚠️ 片段 {chunk_idx + 1} embedding 提取失败（不影响主流程）: {e}")
                
                chunk_result = {
                    "chunk_idx": chunk_idx,
                    "start_time": chunk_start_time,
                    "end_time": chunk_end_time,
                    "annotation": chunk_annotation,
                    "embeddings": chunk_embeddings
                }
                all_chunk_results.append(chunk_result)
                
                for speaker_id, embedding in chunk_embeddings.items():
                    global_key = f"chunk_{chunk_idx}_{speaker_id}"
                    global_embeddings[global_key] = embedding.cpu() if embedding.is_cuda else embedding
                
                del chunk_waveform, chunk_diarization, chunk_annotation
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # 第四步：全局声纹校准（如果没有 embedding，直接合并，不使用校准）
            if len(all_chunk_results) > 1 and global_embeddings:
                logger.info(f"🔗 开始全局声纹校准（共 {len(global_embeddings)} 个 speaker embedding）...")
                id_mapping = global_speaker_calibration(all_chunk_results, global_embeddings, threshold=0.7)
                
                all_segments = Annotation()
                for chunk_result in all_chunk_results:
                    chunk_idx = chunk_result["chunk_idx"]
                    chunk_start_time = chunk_result["start_time"]
                    chunk_annotation = chunk_result["annotation"]
                    
                    for turn, _, speaker in chunk_annotation.itertracks(yield_label=True):
                        # 确保 speaker 不为 None
                        if speaker is None or speaker == "":
                            speaker = f"SPEAKER_{chunk_idx:02d}"
                        
                        chunk_speaker_key = f"chunk_{chunk_idx}_{speaker}"
                        global_speaker_id = id_mapping.get(chunk_speaker_key, speaker)
                        
                        shifted_segment = Segment(
                            turn.start + chunk_start_time,
                            turn.end + chunk_start_time
                        )
                        all_segments[shifted_segment, global_speaker_id] = global_speaker_id
                
                logger.info(f"✅ 全局校准完成，合并 {len(all_chunk_results)} 个片段，共 {len(all_segments)} 个说话人片段")
                return all_segments
            else:
                # 只有一个片段或没有 embedding，直接合并（不使用全局校准）
                if not global_embeddings:
                    logger.info(f"⚠️ 没有提取到 embedding（共 {len(all_chunk_results)} 个片段），跳过全局校准，直接合并")
                else:
                    logger.info(f"⚠️ 只有一个片段，跳过全局校准，直接返回")
                
                all_segments = Annotation()
                for chunk_result in all_chunk_results:
                    chunk_idx = chunk_result["chunk_idx"]
                    chunk_start_time = chunk_result["start_time"]
                    chunk_annotation = chunk_result["annotation"]
                    
                    for turn, _, speaker in chunk_annotation.itertracks(yield_label=True):
                        # 确保 speaker 不为 None
                        if speaker is None or speaker == "":
                            speaker = f"SPEAKER_{chunk_idx:02d}"
                        
                        shifted_segment = Segment(
                            turn.start + chunk_start_time,
                            turn.end + chunk_start_time
                        )
                        all_segments[shifted_segment, speaker] = speaker
                
                logger.info(f"✅ 合并完成，共 {len(all_segments)} 个说话人片段")
                return all_segments
        except Exception as e:
            logger.warning(f"⚠️ VAD 切分处理失败: {e}，回退到固定分段")
            use_vad_smart_chunking = False
    
    # 固定切分 + 全局校准（优化内存使用）
    if not use_vad_smart_chunking or audio_duration >= 3600:
        logger.info(f"⏱️ 音频时长 {audio_duration:.1f}秒，使用固定切分 + 全局声纹校准（优化内存使用）")
        try:
            from pyannote.core import Annotation, Segment
            
            # 减小片段大小以降低内存占用（10-15分钟，避免内存溢出）
            optimized_chunk_duration = min(max_chunk_duration, 900.0)  # 最大 15 分钟（降低内存占用）
            num_chunks = int(audio_duration / optimized_chunk_duration) + 1
            chunk_boundaries = [
                (i * optimized_chunk_duration, min((i + 1) * optimized_chunk_duration, audio_duration))
                for i in range(num_chunks)
            ]
            logger.info(f"✂️ 固定切分为 {len(chunk_boundaries)} 个片段（每段约 {optimized_chunk_duration:.0f}秒，降低内存占用）")
            
            # 第三步：分段推理（优化内存使用）
            all_chunk_results = []
            global_embeddings = {}
            
            logger.info(f"🚀 开始处理 {len(chunk_boundaries)} 个片段（优化内存使用）...")
            
            for chunk_idx, (chunk_start_time, chunk_end_time) in enumerate(chunk_boundaries):
                # 计算采样点
                start_sample = int(chunk_start_time * sample_rate)
                end_sample = int(chunk_end_time * sample_rate)
                chunk_waveform = waveform[:, start_sample:end_sample].clone()  # 使用 clone 避免内存共享
                
                if chunk_waveform.shape[-1] < sample_rate * 0.5:  # 小于0.5秒跳过
                    continue
                
                # 确保 chunk_waveform 在正确的设备上
                if chunk_waveform.device != device:
                    chunk_waveform = chunk_waveform.to(device)
                
                # 清理 GPU 缓存（定期清理，避免内存累积）
                if chunk_idx % 5 == 0 and device.type == "cuda":
                    torch.cuda.empty_cache()
                
                chunk_duration = chunk_end_time - chunk_start_time
                logger.info(f"🔄 处理片段 {chunk_idx + 1}/{len(chunk_boundaries)} ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s, 时长: {chunk_duration:.1f}s) [设备: {chunk_waveform.device}]")
                
                # 处理当前片段
                chunk_diarization = pipeline({"waveform": chunk_waveform, "sample_rate": sample_rate})
                chunk_annotation = _extract_annotation(chunk_diarization)
                
                # 提取该片段的 speaker embeddings（如果失败不影响主流程）
                chunk_embeddings = {}
                try:
                    chunk_embeddings = extract_speaker_embeddings(
                        pipeline, chunk_waveform, sample_rate, chunk_annotation
                    )
                except Exception as e:
                    logger.debug(f"⚠️ 片段 {chunk_idx + 1} embedding 提取失败（不影响主流程）: {e}")
                
                # 存储结果（使用片段前缀避免 ID 冲突）
                chunk_result = {
                    "chunk_idx": chunk_idx,
                    "start_time": chunk_start_time,
                    "end_time": chunk_end_time,
                    "annotation": chunk_annotation,
                    "embeddings": chunk_embeddings
                }
                all_chunk_results.append(chunk_result)
                
                # 添加到全局 embedding 字典（使用唯一 key）
                # 将 embedding 移到 CPU 以节省 GPU 内存
                for speaker_id, embedding in chunk_embeddings.items():
                    global_key = f"chunk_{chunk_idx}_{speaker_id}"
                    global_embeddings[global_key] = embedding.cpu() if embedding.is_cuda else embedding
                
                # 清理当前片段的 GPU 内存
                del chunk_waveform, chunk_diarization, chunk_annotation
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            
            # 第四步：全局声纹校准
            if len(all_chunk_results) > 1 and global_embeddings:
                logger.info("🔗 开始全局声纹校准...")
                id_mapping = global_speaker_calibration(all_chunk_results, global_embeddings, threshold=0.7)
                
                # 合并所有片段，应用校准后的 speaker ID
                all_segments = Annotation()
                for chunk_result in all_chunk_results:
                    chunk_idx = chunk_result["chunk_idx"]
                    chunk_start_time = chunk_result["start_time"]
                    chunk_annotation = chunk_result["annotation"]
                    
                    for turn, _, speaker in chunk_annotation.itertracks(yield_label=True):
                        # 查找校准后的全局 speaker ID
                        chunk_speaker_key = f"chunk_{chunk_idx}_{speaker}"
                        global_speaker_id = id_mapping.get(chunk_speaker_key, speaker)
                        
                        # 创建新的 Segment，时间加上偏移量
                        shifted_segment = Segment(
                            turn.start + chunk_start_time,
                            turn.end + chunk_start_time
                        )
                        all_segments[shifted_segment, global_speaker_id] = global_speaker_id
                
                logger.info(f"✅ 全局校准完成，合并 {len(all_chunk_results)} 个片段")
                return all_segments
            else:
                # 只有一个片段或没有 embedding，直接合并
                all_segments = Annotation()
                for chunk_result in all_chunk_results:
                    chunk_start_time = chunk_result["start_time"]
                    chunk_annotation = chunk_result["annotation"]
                    
                    for turn, _, speaker in chunk_annotation.itertracks(yield_label=True):
                        shifted_segment = Segment(
                            turn.start + chunk_start_time,
                            turn.end + chunk_start_time
                        )
                        all_segments[shifted_segment, speaker] = speaker
                
                return all_segments
                
        except Exception as e:
            logger.warning(f"⚠️ VAD 智能分段失败，回退到固定分段: {e}")
            # 回退到原来的固定分段逻辑
            use_vad_smart_chunking = False
    
    # 回退：固定时长分段（原逻辑）
    logger.info(f"⏱️ 音频时长 {audio_duration:.1f}秒，使用固定分段处理（每段 {max_chunk_duration}秒）")
    try:
        from pyannote.core import Annotation, Segment
        
        all_segments = Annotation()
        num_chunks = int(audio_duration / max_chunk_duration) + 1
        
        for chunk_idx in range(num_chunks):
            chunk_start_time = chunk_idx * max_chunk_duration
            chunk_end_time = min((chunk_idx + 1) * max_chunk_duration, audio_duration)
            
            start_sample = int(chunk_start_time * sample_rate)
            end_sample = int(chunk_end_time * sample_rate)
            chunk_waveform = waveform[:, start_sample:end_sample]
            
            if chunk_waveform.shape[-1] < sample_rate * 0.5:
                continue
            
            if chunk_waveform.device != device:
                chunk_waveform = chunk_waveform.to(device)
            
            logger.info(f"🔄 处理片段 {chunk_idx + 1}/{num_chunks} ({chunk_start_time:.1f}s - {chunk_end_time:.1f}s) [设备: {chunk_waveform.device}]")
            
            chunk_diarization = pipeline({"waveform": chunk_waveform, "sample_rate": sample_rate})
            chunk_annotation = _extract_annotation(chunk_diarization)
            
            for turn, _, speaker in chunk_annotation.itertracks(yield_label=True):
                # 确保 speaker 不为 None 或空
                if speaker is None or speaker == "":
                    speaker = f"SPEAKER_{chunk_idx:02d}"  # 使用片段索引作为默认 speaker
                
                shifted_segment = Segment(turn.start + chunk_start_time, turn.end + chunk_start_time)
                all_segments[shifted_segment, speaker] = speaker
        
        return all_segments
    except ImportError:
        logger.warning("⚠️ 无法导入 pyannote.core，降级为全量处理")
        return pipeline({"waveform": waveform, "sample_rate": sample_rate})


def get_pyannote_pipeline(use_auth_token: Optional[str] = None):
    """
    获取 Pyannote pipeline（优先使用项目本地 models/ 目录，强制离线，避免任何联网）
    """
    global _pipeline_cache
    
    if _pipeline_cache is not None:
        return _pipeline_cache
    
    if not PYANNOTE_AVAILABLE:
        return None
    
    # 统一 token（兼容历史命名 use_auth_token）
    hf_token = use_auth_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or None

    # 强制离线：彻底禁止 huggingface_hub 发起任何请求
    # 说明：如果本地模型不完整，会直接报缺文件，而不是偷偷联网补齐
    # 用强制覆盖而不是 setdefault，避免外部环境提前设置为 0 导致失效
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def _safe_from_pretrained(model_ref: str):
        """兼容不同版本 pyannote.audio / huggingface 参数名（token vs use_auth_token）"""
        # 本地路径加载：不要传 token/use_auth_token（避免参数不兼容，也避免触发 hub 逻辑）
        if Path(model_ref).exists():
            # 这里如果失败，应该直接报错（不要再回退到 token/use_auth_token，否则会误触发 hub 逻辑）
            return Pipeline.from_pretrained(model_ref)

        # 远端（或 cache）加载：根据参数名做兼容
        try:
            if hf_token:
                return Pipeline.from_pretrained(model_ref, token=hf_token)
            return Pipeline.from_pretrained(model_ref)
        except TypeError as e:
            # 只有当确实是“token 参数不被支持”时，才回退 use_auth_token
            msg = str(e)
            if hf_token and ("token" in msg or "unexpected keyword argument" in msg):
                return Pipeline.from_pretrained(model_ref, use_auth_token=hf_token)
            raise

    try:
        # 1) 优先从项目本地 models/ 目录加载（推荐、离线可用）
        project_root = Path(__file__).resolve().parent.parent
        local_diar_dir = project_root / "models" / "pyannote_diarization"
        local_cfg = local_diar_dir / "config.yaml"
        local_offline_cfg = local_diar_dir / "offline_config.yaml"
        local_seg_dir = project_root / "models" / "pyannote_segmentation"
        local_emb_dir = project_root / "models" / "pyannote_wespeaker"

        if local_diar_dir.exists() and local_cfg.exists():
            cfg_source = local_offline_cfg if local_offline_cfg.exists() else local_cfg
            logger.info(f"✅ 使用本地 Pyannote 配置加载（离线）: {cfg_source}")

            # 检查子模型目录是否存在
            if not local_seg_dir.exists():
                logger.error(f"❌ 缺少本地 segmentation 模型目录: {local_seg_dir}")
                return None
            if not local_emb_dir.exists():
                logger.error(f"❌ 缺少本地 embedding 模型目录: {local_emb_dir}")
                return None

            # 动态改写 config.yaml：把 segmentation/embedding 指到本地目录（而不是 HuggingFace ID 或具体 bin 文件）
            try:
                import yaml  # PyYAML
            except Exception as e:
                logger.error(f"❌ 缺少 PyYAML，无法改写 config.yaml: {e}")
                logger.error("   请安装: pip install PyYAML")
                return None

            original_cfg_bytes = local_cfg.read_bytes()
            try:
                cfg_obj = yaml.safe_load(cfg_source.read_text(encoding="utf-8"))
                if not isinstance(cfg_obj, dict):
                    raise ValueError("config.yaml 解析结果不是 dict")

                pipeline_section = cfg_obj.setdefault("pipeline", {})
                params = pipeline_section.setdefault("params", {})

                # 旧版 offline_config 里可能把 clustering 配成 dict，但你这版 pyannote.audio 期望是字符串 key。
                # 这里直接删掉 clustering，让 pipeline 使用自己的默认聚类配置（我们只控制模型路径和 PLDA）。
                # 注意：clustering 必须是字符串（如 "centroid"），不能是 dict
                if isinstance(params.get("clustering"), dict):
                    logger.info("ℹ️ 检测到 dict 类型 clustering 配置，已移除以使用默认聚类策略")
                    params.pop("clustering", None)
                # 确保 clustering 不是 dict（如果存在，必须是字符串）
                if "clustering" in params and isinstance(params["clustering"], dict):
                    params.pop("clustering", None)
                    logger.info("ℹ️ 已移除 dict 类型 clustering 配置")

                # 关键：显式指定 PLDA 资源来源。
                # 说明：当前 pyannote.audio 版本不接受 plda=None（只接受 str/dict），
                # 且 SpeakerDiarization 默认会加载 `pyannote/speaker-diarization-community-1`
                # 并在缺 cache 时尝试联网下载 plda/xvec_transform.npz。
                # 我们在强制离线模式下优先使用项目本地目录，否则退回到 HF cache（不会联网）。
                local_plda_candidates = [
                    project_root / "models" / "pyannote_speaker_diarization_community_1",
                    project_root / "models" / "speaker-diarization-community-1",
                    project_root / "models" / "pyannote_plda",
                ]
                plda_ref = None
                for cand in local_plda_candidates:
                    # community-1 可能是 `xvec_transform.npz` 放在根目录，也可能在 `plda/` 子目录
                    xvec_root = cand / "xvec_transform.npz"
                    plda_root = cand / "plda.npz"
                    xvec_sub = cand / "plda" / "xvec_transform.npz"
                    plda_sub = cand / "plda" / "plda.npz"

                    if xvec_root.exists() and plda_root.exists():
                        # 直接使用目录路径（pyannote 会在目录根下找这两个 npz）
                        plda_ref = str(cand.resolve())
                        break
                    if xvec_sub.exists() and plda_sub.exists():
                        # 兼容：有的 repo 把 npz 放在 plda/ 子目录里，但部分 pyannote 版本只会在根目录找。
                        # 这里将两个 npz “展开/复制” 到一个扁平目录中，再用字符串路径指向该目录。
                        flat_dir = project_root / "models" / "_pyannote_plda_flat"
                        flat_dir.mkdir(parents=True, exist_ok=True)
                        flat_xvec = flat_dir / "xvec_transform.npz"
                        flat_plda = flat_dir / "plda.npz"
                        try:
                            shutil.copy2(xvec_sub, flat_xvec)
                            shutil.copy2(plda_sub, flat_plda)
                            logger.info(f"✅ 已展开 PLDA 文件到扁平目录: {flat_dir}")
                        except Exception as e:
                            logger.error(f"❌ 展开 PLDA 文件失败: {e}")
                            return None
                        plda_ref = str(flat_dir.resolve())
                        break
                if plda_ref:
                    params["plda"] = plda_ref
                    logger.info(f"✅ 使用本地 PLDA 资源: {plda_ref}")
                else:
                    # 使用 HF cache（离线环境下不会下载；若 cache 不存在，会明确报缺文件）
                    params["plda"] = "pyannote/speaker-diarization-community-1"
                    logger.warning("⚠️ 未找到项目本地 PLDA 资源目录，将使用 HF cache: pyannote/speaker-diarization-community-1（已强制离线，不会联网）")

                # 关键：指向本地目录（目录里包含 config.yaml / pytorch_model.bin 等）
                params["segmentation"] = str(local_seg_dir.resolve())
                params["embedding"] = str(local_emb_dir.resolve())
                
                # 性能优化：在 config.yaml 中设置参数（在 pipeline 加载前设置）
                # 根据警告信息，pipeline 会自动实例化，我们需要在 config 中设置参数
                # 方法：将 segmentation 从字符串路径改为字典配置，包含模型路径和参数
                seg_path = params.get("segmentation", str(local_seg_dir.resolve()))
                # 注意：segmentation 必须保持为字符串路径或包含 checkpoint 的字典
                # Pyannote 的 get_model 函数期望 checkpoint 参数，不是 model
                if isinstance(seg_path, str):
                    # 保持字符串路径，min_duration 参数将在 pipeline 实例化后设置
                    params["segmentation"] = seg_path
                    logger.info("⚙️ segmentation 保持为路径格式，min_duration 参数将在 pipeline 实例化后设置")
                elif isinstance(seg_path, dict):
                    # 如果已经是字典，确保使用 checkpoint 而不是 model
                    if "model" in seg_path and "checkpoint" not in seg_path:
                        seg_path["checkpoint"] = seg_path.pop("model")
                        logger.info("⚙️ 已将 segmentation.model 改为 checkpoint")
                    # 添加性能优化参数
                    if "checkpoint" in seg_path:
                        seg_path.setdefault("min_duration_on", 0.5)
                        seg_path.setdefault("min_duration_off", 0.5)
                        logger.info("⚙️ 已更新 segmentation 参数: min_duration_on=0.5, min_duration_off=0.5")
                    else:
                        logger.warning("⚠️ segmentation 字典中缺少 checkpoint 参数，将使用字符串路径")
                        params["segmentation"] = str(local_seg_dir.resolve())
                
                # 注意：clustering 必须是字符串（如 "centroid"），不能是 dict
                # 如果需要优化聚类，应该在 pipeline 实例化后通过其他方式设置
                # 这里不设置 clustering 参数，让 pipeline 使用默认值

                # 写入到 models/pyannote_diarization/config.yaml（临时覆盖，加载后恢复）
                local_cfg.write_text(yaml.safe_dump(cfg_obj, sort_keys=False, allow_unicode=True), encoding="utf-8")
                logger.info("✅ 已将 segmentation/embedding 映射到本地 models 目录，并设置性能优化参数")

                pipeline = _safe_from_pretrained(str(local_diar_dir))
                
                # 关键：将 pipeline 移动到 GPU（如果可用）
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if torch.cuda.is_available():
                    pipeline = pipeline.to(device)
                    logger.info(f"✅ Pipeline 已移动到 GPU: {device}")
                    # 显示 GPU 信息
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
                    logger.info(f"🎮 GPU 信息: {gpu_name}, 显存: {gpu_memory:.1f}GB")
                else:
                    logger.warning("⚠️ CUDA 不可用，使用 CPU（速度较慢）")
                
                # 性能优化：在 pipeline 实例化后设置参数
                # 根据警告信息，pipeline 会自动实例化，我们需要在实例化后设置参数
                try:
                    # 方法1：尝试通过 pipeline 的内部属性设置参数
                    # Pyannote 3.x 的 pipeline 可能有不同的内部结构
                    if hasattr(pipeline, "segmentation"):
                        seg = pipeline.segmentation
                        
                        # 尝试多种方式设置参数
                        attrs_to_try = [
                            ("min_duration_on", 0.5),
                            ("min_duration_off", 0.5),
                        ]
                        
                        for attr_name, attr_value in attrs_to_try:
                            if hasattr(seg, attr_name):
                                setattr(seg, attr_name, attr_value)
                                logger.info(f"⚙️ 已设置 segmentation.{attr_name}: {attr_value}秒")
                            # 尝试通过内部对象设置
                            elif hasattr(seg, "_segmentation"):
                                inner_seg = seg._segmentation
                                if hasattr(inner_seg, attr_name):
                                    setattr(inner_seg, attr_name, attr_value)
                                    logger.info(f"⚙️ 已设置 segmentation._segmentation.{attr_name}: {attr_value}秒")
                            # 尝试通过 params 设置
                            elif hasattr(seg, "params") and isinstance(seg.params, dict):
                                seg.params[attr_name] = attr_value
                                logger.info(f"⚙️ 已通过 segmentation.params 设置 {attr_name}: {attr_value}秒")
                    
                    # 方法2：尝试设置批处理大小和优化聚类参数
                    if hasattr(pipeline, "batch_size"):
                        pipeline.batch_size = 32
                        logger.info(f"⚙️ 已设置 pipeline.batch_size: 32")
                    elif hasattr(pipeline, "segmentation") and hasattr(pipeline.segmentation, "batch_size"):
                        pipeline.segmentation.batch_size = 32
                        logger.info(f"⚙️ 已设置 segmentation.batch_size: 32")
                    
                    # 方法3：优化聚类参数（减少计算量）
                    if hasattr(pipeline, "clustering"):
                        clustering = pipeline.clustering
                        if hasattr(clustering, "threshold"):
                            clustering.threshold = 0.7  # 提高阈值，减少片段
                            logger.info("⚙️ 已设置 clustering.threshold: 0.7（减少片段数量）")
                except Exception as e:
                    logger.debug(f"⚠️ 设置 pipeline 参数失败（将使用调用时传递参数）: {e}")
                
                _pipeline_cache = pipeline
                logger.info("✅ 本地 Pyannote pipeline 加载成功（全程离线，已优化性能参数）")
                return pipeline
            except Exception as e:
                logger.error(f"❌ 本地 Pyannote pipeline 加载失败: {e}", exc_info=True)
                return None
            finally:
                # 恢复原 config.yaml，避免污染仓库文件
                try:
                    local_cfg.write_bytes(original_cfg_bytes)
                except Exception:
                    pass

        # 2) 如果没找到本地模型目录：明确提示（不再尝试联网）
        logger.error("❌ 未找到本地 Pyannote 模型目录或配置文件（models/pyannote_diarization/config.yaml）")
        logger.error(f"   期望路径: {local_diar_dir}")
        return None

    except Exception as e:
        logger.error(f"❌ Pyannote 初始化错误: {e}", exc_info=True)
        return None
    
    return None


def perform_pyannote_diarization(
    audio_path: str,
    transcript: List[Dict],
    use_auth_token: Optional[str] = None
) -> List[Dict]:
    """
    使用 Pyannote 进行说话人分离
    """
    if not PYANNOTE_AVAILABLE:
        logger.error("❌ Pyannote.audio 未安装")
        return transcript

    try:
        logger.info("🎤 使用 Pyannote.audio 进行说话人分离...")
        
        # 1. 获取 pipeline (逻辑都在 get_pyannote_pipeline 里处理好了)
        pipeline = get_pyannote_pipeline(use_auth_token)
        
        if pipeline is None:
            logger.error("❌ 无法加载 Pyannote pipeline，跳过分离步骤")
            # 降级处理：全标记为 0
            for item in transcript:
                if 'speaker_id' not in item:
                    item['speaker_id'] = "0"
            return transcript

        # 2. 处理音频（支持 URL：如果是 http(s)，在服务器端先下载到临时文件再推理）
        logger.info(f"📂 处理音频文件: {audio_path}")
        tmp_path = None
        try:
            if isinstance(audio_path, str) and audio_path.startswith(("http://", "https://")):
                import requests
                import tempfile

                logger.info(f"🔗 检测到音频 URL，正在服务器端下载到临时文件: {audio_path}")
                resp = requests.get(audio_path, timeout=300, stream=True)
                resp.raise_for_status()

                suffix = Path(audio_path).suffix or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    for chunk in resp.iter_content(chunk_size=8192):
                        if chunk:
                            tmp.write(chunk)
                    tmp_path = tmp.name
                logger.info(f"✅ 音频已下载到临时文件: {tmp_path}")
                audio_path = tmp_path

            # 手动解码音频，绕过 pyannote 内部 AudioDecoder
            if not os.path.exists(audio_path):
                raise FileNotFoundError(f"Audio file not found: {audio_path}")

            # 如果格式不支持（如 M4A），使用 ffmpeg 转换
            converted_audio_path = None
            try:
                # 尝试直接读取
                data, sample_rate = sf.read(audio_path)
            except Exception as e:
                # 如果 soundfile 不支持该格式，使用 ffmpeg 转换为 WAV
                logger.info(f"⚠️ soundfile 不支持该格式，使用 ffmpeg 转换: {audio_path}")
                try:
                    # 使用 ffmpeg 转换为 WAV
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
                        converted_audio_path = tmp_wav.name
                    
                    cmd = [
                        "ffmpeg", "-i", audio_path,
                        "-ac", "1",  # 单声道
                        "-ar", "16000",  # 16kHz 采样率
                        "-f", "wav",
                        "-y",  # 覆盖输出文件
                        converted_audio_path
                    ]
                    
                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        timeout=60
                    )
                    
                    # 读取转换后的 WAV 文件
                    data, sample_rate = sf.read(converted_audio_path)
                    logger.info(f"✅ 音频格式转换成功: {converted_audio_path}")
                except subprocess.CalledProcessError as ffmpeg_error:
                    error_msg = ffmpeg_error.stderr.decode() if ffmpeg_error.stderr else str(ffmpeg_error)
                    raise RuntimeError(f"ffmpeg 转换失败: {error_msg}") from e
                except FileNotFoundError:
                    raise RuntimeError("ffmpeg 未安装，无法处理 M4A 等格式。请安装: apt-get install ffmpeg 或 conda install ffmpeg") from e
            if data.ndim == 1:
                data = data[None, :]
            else:
                data = data.T

            waveform = torch.tensor(data, dtype=torch.float32)
            
            # 确保 waveform 在正确的设备上（与 pipeline 一致）
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if waveform.device != device:
                waveform = waveform.to(device)
                logger.info(f"🔄 已将 waveform 移动到设备: {device}")
            
            # 使用公共函数处理音频（支持长音频分段处理）
            diarization = process_audio_with_pipeline(pipeline, waveform, sample_rate)
        finally:
            # 清理 URL 下载产生的临时文件
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                    logger.info(f"🧹 已清理 Pyannote 临时音频文件: {tmp_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理 Pyannote 临时音频文件失败: {tmp_path}, {e}")
            
            # 清理格式转换产生的临时文件
            if 'converted_audio_path' in locals() and converted_audio_path and os.path.exists(converted_audio_path):
                try:
                    os.remove(converted_audio_path)
                    logger.debug(f"🧹 已清理格式转换临时文件: {converted_audio_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理格式转换临时文件失败: {e}")
        
        # 3. 兼容不同版本输出，拿到真正的 Annotation
        annotation = _extract_annotation(diarization)

        # 4. 构建说话人时间映射
        speaker_segments = []
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            speaker_segments.append({
                'start_time': turn.start,
                'end_time': turn.end,
                'speaker_id': speaker
            })
        
        logger.info(f"✅ Pyannote 识别出 {len(set(s['speaker_id'] for s in speaker_segments))} 个说话人")

        # 5. 将说话人信息映射到 transcript (后面的代码保持不变)
        for item in transcript:
            item_start = item.get('start_time', 0)
            item_end = item.get('end_time', 0)
            
            # ... (这部分的对齐逻辑你原来写的没问题，保留) ...
            matched_speaker = None
            max_overlap = 0
            
            for seg in speaker_segments:
                seg_start = seg['start_time']
                seg_end = seg['end_time']
                overlap_start = max(item_start, seg_start)
                overlap_end = min(item_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                
                item_duration = item_end - item_start
                if item_duration > 0 and overlap / item_duration > 0.5:
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matched_speaker = seg['speaker_id']
            
            item['speaker_id'] = matched_speaker if matched_speaker else "SPEAKER_00"

        # 5. 规范化 ID
        speaker_id_map = {}
        speaker_counter = 0
        for item in transcript:
            original_id = item.get('speaker_id', 'SPEAKER_00')
            if original_id not in speaker_id_map:
                speaker_id_map[original_id] = str(speaker_counter)
                speaker_counter += 1
            item['speaker_id'] = speaker_id_map[original_id]
        
        return transcript
        
    except Exception as e:
        logger.error(f"❌ Pyannote 说话人分离失败: {e}", exc_info=True)
        for item in transcript:
            item['speaker_id'] = "0"
        return transcript
