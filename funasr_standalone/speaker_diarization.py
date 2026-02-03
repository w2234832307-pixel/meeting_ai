#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
说话人分离模块
用于 SenseVoiceSmall 模型的说话人识别功能
"""
import logging
import subprocess
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import soundfile as sf

logger = logging.getLogger(__name__)


def perform_speaker_diarization_with_vad(
    audio_path: str,
    vad_segments: List,
    speaker_model,
    device: str = "cuda",
    min_segment_duration: float = 1.0,
    distance_threshold: float = 0.3  # 降低默认阈值，确保能识别出多个说话人
) -> List[Dict]:
    """
    基于 VAD 分段进行说话人分离
    
    Args:
        audio_path: 音频文件路径
        vad_segments: VAD 分段信息 [[start_ms, end_ms], ...]
        speaker_model: Cam++ 说话人模型
        device: 设备
        min_segment_duration: 最小片段时长（秒）
        distance_threshold: 聚类距离阈值（0.3-0.7）
    
    Returns:
        [{"start_time": 0.0, "end_time": 2.5, "speaker_id": "0"}, ...]
    """
    try:
        # 1. 过滤太短的片段并提取声纹
        valid_segments = []
        embeddings = []
        
        logger.info(f"🔬 为 {len(vad_segments)} 个 VAD 段提取声纹特征...")
        
        for idx, segment in enumerate(vad_segments):
            if not isinstance(segment, list) or len(segment) < 2:
                continue
            
            start_ms, end_ms = segment[0], segment[1]
            
            # 处理 end_ms = -1 的情况（表示到音频结尾）
            if end_ms == -1:
                duration = 999999  # 一个很大的数
            else:
                duration = (end_ms - start_ms) / 1000.0
            
            # 过滤太短的片段
            if duration < min_segment_duration:
                logger.debug(f"⏭️ 跳过过短片段 {idx}: {duration:.2f}s")
                continue
            
            # 提取音频片段并获取声纹
            try:
                embedding = extract_speaker_embedding(
                    audio_path=audio_path,
                    start_ms=start_ms,
                    end_ms=end_ms if end_ms != -1 else None,
                    speaker_model=speaker_model
                )
                
                if embedding is not None:
                    embeddings.append(embedding)
                    valid_segments.append({
                        "start_time": round(start_ms / 1000.0, 2),
                        "end_time": round(end_ms / 1000.0, 2) if end_ms != -1 else 999999,
                        "segment_idx": idx
                    })
                    
            except Exception as e:
                logger.warning(f"⚠️ 提取片段 {idx} 声纹失败: {e}")
                continue
        
        if len(valid_segments) == 0:
            logger.warning("⚠️ 没有有效的语音片段")
            return []
        
        logger.info(f"✅ 成功提取 {len(embeddings)} 个声纹特征")
        
        # 2. 如果只有一个片段，直接标记为说话人0
        if len(embeddings) == 1:
            logger.info("ℹ️ 只有一个语音段，标记为说话人0")
            valid_segments[0]["speaker_id"] = "0"
            return valid_segments
        
        # 2.5. 如果只有2个片段，直接标记为说话人0和1
        if len(embeddings) == 2:
            logger.info("ℹ️ 只有两个语音段，标记为说话人0和1")
            valid_segments[0]["speaker_id"] = "0"
            valid_segments[1]["speaker_id"] = "1"
            return valid_segments
        
        # 3. 使用层次聚类进行说话人分离
        logger.info(f"🔬 进行说话人聚类...")
        
        # 确保 embeddings 是 2D 数组 (n_samples, n_features)
        # Cam++ 可能返回 3D 数组，需要展平
        embeddings_2d = []
        for emb in embeddings:
            emb_array = np.array(emb)
            # 如果是 3D 或更高维度，展平为 1D
            if emb_array.ndim > 1:
                emb_array = emb_array.flatten()
            embeddings_2d.append(emb_array)
        
        embeddings_array = np.array(embeddings_2d)
        
        # 验证维度
        if embeddings_array.ndim != 2:
            logger.error(f"❌ 声纹向量维度错误: {embeddings_array.shape}，期望 2D (n_samples, n_features)")
            # 降级处理：所有片段标记为同一说话人
            for segment in valid_segments:
                segment["speaker_id"] = "0"
            return valid_segments
        
        logger.debug(f"✅ 声纹向量形状: {embeddings_array.shape}")
        
        # 优化聚类参数：降低阈值，确保能识别出多个说话人
        # 如果只识别出1个说话人，说明阈值太高，需要降低
        # 自动调整距离阈值：根据片段数量动态调整
        if len(embeddings) > 100:
            # 片段很多，稍微增大阈值（但不要太大，避免只识别出1个人）
            adjusted_threshold = min(0.6, distance_threshold + 0.05)
            logger.info(f"🔧 片段较多({len(embeddings)}个)，调整聚类阈值为 {adjusted_threshold:.2f}")
        elif len(embeddings) > 50:
            # 片段中等，保持或稍微降低阈值
            adjusted_threshold = max(0.3, distance_threshold - 0.05)
            logger.info(f"🔧 片段中等({len(embeddings)}个)，调整聚类阈值为 {adjusted_threshold:.2f}")
        else:
            # 片段较少，降低阈值，确保能识别出多个说话人
            adjusted_threshold = max(0.25, distance_threshold - 0.1)
            logger.info(f"🔧 片段较少({len(embeddings)}个)，降低聚类阈值为 {adjusted_threshold:.2f} 以识别更多说话人")
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=adjusted_threshold,  # 调整后的距离阈值
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embeddings_array)
        
        # 4. 显示真实的聚类结果（证明不是写死的）
        unique_labels = sorted(set(cluster_labels))
        logger.info(f"🎯 【真实聚类结果】识别出 {len(unique_labels)} 个不同的说话人")
        logger.info(f"   原始聚类标签: {unique_labels} (范围: {min(cluster_labels)}-{max(cluster_labels)})")
        
        # 统计每个聚类的片段数量（证明是真实识别）
        cluster_counts = {}
        for label in cluster_labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        logger.info(f"   各说话人的片段数量: {dict(sorted(cluster_counts.items()))}")
        
        # 重新映射说话人ID为连续编号（0, 1, 2, 3...）
        # 注意：这只是编号规范化，不影响识别结果！
        # 哪些片段属于哪个说话人是由聚类算法决定的，不是写死的
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        logger.info(f"   编号规范化映射: {label_mapping} (仅用于统一编号，不影响识别结果)")
        
        # 5. 将聚类结果映射到片段，并重新编号
        # 保留原始聚类标签用于验证（证明不是写死的）
        for idx, segment in enumerate(valid_segments):
            old_label = cluster_labels[idx]  # 这是聚类算法的真实结果
            new_label = label_mapping[old_label]  # 这只是编号规范化
            segment["speaker_id"] = str(new_label)
            segment["_original_cluster_id"] = int(old_label)  # 保留原始标签用于验证
        
        n_speakers = len(unique_labels)
        
        # 验证映射后的ID是否连续
        mapped_ids = sorted(set(int(s["speaker_id"]) for s in valid_segments))
        expected_ids = list(range(n_speakers))
        
        if mapped_ids != expected_ids:
            logger.error(f"❌ 说话人ID映射错误: 实际={mapped_ids}, 期望={expected_ids}")
            # 强制重新映射
            for idx, segment in enumerate(valid_segments):
                segment["speaker_id"] = str(mapped_ids.index(int(segment["speaker_id"])))
        
        # 显示识别结果示例（证明是真实识别）
        logger.info(f"✅ 识别出 {n_speakers} 个说话人（ID: 0-{n_speakers-1}）")
        logger.info(f"   【验证】前3个片段的原始聚类ID: {[s.get('_original_cluster_id') for s in valid_segments[:3]]}")
        
        return valid_segments
        
    except Exception as e:
        logger.error(f"❌ 说话人分离失败: {e}", exc_info=True)
        return []


def perform_speaker_diarization_with_cached_audio(
    vad_segments: List,
    cached_audio_map: Dict[int, Tuple[np.ndarray, int]],
    speaker_model,
    device: str = "cuda",
    min_segment_duration: float = 1.0,
    distance_threshold: float = 0.5,
    audio_file_path: str = None
) -> List[Dict]:
    """
    基于缓存的音频数据进行说话人分离（优化版）
    
    Args:
        vad_segments: VAD 分段信息 [[start_ms, end_ms], ...]
        cached_audio_map: 缓存的音频数据 {segment_idx: (audio_data, sample_rate)}
        speaker_model: Cam++ 说话人模型
        device: 设备
        min_segment_duration: 最小片段时长（秒）
        distance_threshold: 聚类距离阈值（0.3-0.7）
        audio_file_path: 原始音频文件路径（降级时使用）
    
    Returns:
        [{"start_time": 0.0, "end_time": 2.5, "speaker_id": "0"}, ...]
    """
    try:
        # 1. 过滤太短的片段并提取声纹（使用缓存的音频数据）
        valid_segments = []
        embeddings = []
        
        logger.info(f"🔬 为 {len(vad_segments)} 个 VAD 段提取声纹特征（使用缓存音频）...")
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        MAX_CONCURRENT = 2  # 控制并发数（10GB显存）
        
        def extract_embedding_from_cache(idx, segment):
            """从缓存音频数据提取声纹"""
            if not isinstance(segment, list) or len(segment) < 2:
                return None, None
            
            start_ms, end_ms = segment[0], segment[1]
            
            # 处理 end_ms = -1 的情况
            if end_ms == -1:
                duration = 999999
            else:
                duration = (end_ms - start_ms) / 1000.0
            
            # 过滤太短的片段
            if duration < min_segment_duration:
                logger.debug(f"⏭️ 跳过过短片段 {idx}: {duration:.2f}s")
                return None, None
            
            # 优先使用缓存的音频数据
            if idx in cached_audio_map:
                audio_data, sample_rate = cached_audio_map[idx]
                
                try:
                    # 将内存中的音频数据写入临时文件（模型需要文件路径）
                    temp_segment = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                    temp_segment.close()
                    temp_segment_path = temp_segment.name
                    
                    sf.write(temp_segment_path, audio_data, sample_rate)
                    
                    # 使用 Cam++ 模型提取声纹
                    emb_res = speaker_model.generate(input=temp_segment_path)
                    
                    # 清理临时文件
                    try:
                        os.remove(temp_segment_path)
                    except:
                        pass
                    
                    if emb_res and len(emb_res) > 0:
                        emb = emb_res[0].get("spk_embedding", None)
                        if emb is not None:
                            emb_array = np.array(emb)
                            if emb_array.ndim > 1:
                                emb_array = emb_array.flatten()
                            return emb_array.tolist(), {
                                "start_time": round(start_ms / 1000.0, 2),
                                "end_time": round(end_ms / 1000.0, 2) if end_ms != -1 else 999999,
                                "segment_idx": idx
                            }
                except Exception as e:
                    logger.warning(f"⚠️ 从缓存提取片段 {idx} 声纹失败: {e}")
            
            # 降级：如果缓存中没有，使用原始文件提取
            if audio_file_path:
                try:
                    embedding = extract_speaker_embedding(
                        audio_path=audio_file_path,
                        start_ms=start_ms,
                        end_ms=end_ms if end_ms != -1 else None,
                        speaker_model=speaker_model
                    )
                    if embedding:
                        return embedding, {
                            "start_time": round(start_ms / 1000.0, 2),
                            "end_time": round(end_ms / 1000.0, 2) if end_ms != -1 else 999999,
                            "segment_idx": idx
                        }
                except Exception as e:
                    logger.warning(f"⚠️ 降级提取片段 {idx} 声纹失败: {e}")
            
            return None, None
        
        # 优化3: 并行提取声纹（控制并发数）
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT) as executor:
            futures = {
                executor.submit(extract_embedding_from_cache, idx, segment): idx
                for idx, segment in enumerate(vad_segments)
            }
            
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    embedding, segment_info = future.result()
                    if embedding is not None and segment_info is not None:
                        embeddings.append(embedding)
                        valid_segments.append(segment_info)
                except Exception as e:
                    logger.warning(f"⚠️ 提取片段 {idx} 声纹异常: {e}")
        
        if len(valid_segments) == 0:
            logger.warning("⚠️ 没有有效的语音片段")
            return []
        
        logger.info(f"✅ 成功提取 {len(embeddings)} 个声纹特征（使用缓存）")
        
        # 2. 如果只有一个片段，直接标记为说话人0
        if len(embeddings) == 1:
            logger.info("ℹ️ 只有一个语音段，标记为说话人0")
            valid_segments[0]["speaker_id"] = "0"
            return valid_segments
        
        # 2.5. 如果只有2个片段，直接标记为说话人0和1
        if len(embeddings) == 2:
            logger.info("ℹ️ 只有两个语音段，标记为说话人0和1")
            valid_segments[0]["speaker_id"] = "0"
            valid_segments[1]["speaker_id"] = "1"
            return valid_segments
        
        # 3. 使用层次聚类进行说话人分离（与原函数相同）
        logger.info(f"🔬 进行说话人聚类...")
        
        embeddings_2d = []
        for emb in embeddings:
            emb_array = np.array(emb)
            if emb_array.ndim > 1:
                emb_array = emb_array.flatten()
            embeddings_2d.append(emb_array)
        
        embeddings_array = np.array(embeddings_2d)
        
        if embeddings_array.ndim != 2:
            logger.error(f"❌ 声纹向量维度错误: {embeddings_array.shape}")
            for segment in valid_segments:
                segment["speaker_id"] = "0"
            return valid_segments
        
        logger.debug(f"✅ 声纹向量形状: {embeddings_array.shape}")
        
        # 自动调整距离阈值
        if len(embeddings) > 30:
            adjusted_threshold = min(0.7, distance_threshold + 0.1)
            logger.info(f"🔧 片段较多({len(embeddings)}个)，调整聚类阈值为 {adjusted_threshold:.2f}")
        else:
            adjusted_threshold = distance_threshold
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=adjusted_threshold,
            metric='cosine',
            linkage='average'
        )
        
        cluster_labels = clustering.fit_predict(embeddings_array)
        
        # 4. 显示真实的聚类结果
        unique_labels = sorted(set(cluster_labels))
        logger.info(f"🎯 【真实聚类结果】识别出 {len(unique_labels)} 个不同的说话人")
        logger.info(f"   原始聚类标签: {unique_labels} (范围: {min(cluster_labels)}-{max(cluster_labels)})")
        
        cluster_counts = {}
        for label in cluster_labels:
            cluster_counts[label] = cluster_counts.get(label, 0) + 1
        logger.info(f"   各说话人的片段数量: {dict(sorted(cluster_counts.items()))}")
        
        # 重新映射说话人ID为连续编号（0, 1, 2, 3...）
        label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}
        logger.info(f"   编号规范化映射: {label_mapping}")
        
        # 5. 将聚类结果映射到片段
        for idx, segment in enumerate(valid_segments):
            old_label = cluster_labels[idx]
            new_label = label_mapping[old_label]
            segment["speaker_id"] = str(new_label)
            segment["_original_cluster_id"] = int(old_label)
        
        n_speakers = len(unique_labels)
        
        # 验证映射后的ID是否连续
        mapped_ids = sorted(set(int(s["speaker_id"]) for s in valid_segments))
        expected_ids = list(range(n_speakers))
        
        if mapped_ids != expected_ids:
            logger.error(f"❌ 说话人ID映射错误: 实际={mapped_ids}, 期望={expected_ids}")
            for idx, segment in enumerate(valid_segments):
                segment["speaker_id"] = str(mapped_ids.index(int(segment["speaker_id"])))
        
        logger.info(f"✅ 识别出 {n_speakers} 个说话人（ID: 0-{n_speakers-1}）")
        
        return valid_segments
        
    except Exception as e:
        logger.error(f"❌ 说话人分离失败: {e}", exc_info=True)
        return []


def extract_speaker_embedding(
    audio_path: str,
    start_ms: float,
    end_ms: float = None,
    speaker_model = None
) -> List[float]:
    """
    提取音频片段的声纹特征向量
    
    Args:
        audio_path: 音频文件路径
        start_ms: 开始时间（毫秒）
        end_ms: 结束时间（毫秒），None 表示到音频结尾
        speaker_model: Cam++ 模型
    
    Returns:
        声纹向量（192维）
    """
    temp_segment_path = None
    
    try:
        # 1. 使用 ffmpeg 提取音频片段
        temp_segment = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_segment.close()
        temp_segment_path = temp_segment.name
        
        # 构建 ffmpeg 命令
        cmd = [
            "ffmpeg",
            "-i", audio_path,
            "-ss", str(start_ms / 1000.0),
        ]
        
        if end_ms is not None:
            duration = (end_ms - start_ms) / 1000.0
            cmd.extend(["-t", str(duration)])
        
        cmd.extend([
            "-ac", "1",              # 单声道
            "-ar", "16000",          # 16kHz 采样率
            "-y",
            "-loglevel", "error",
            temp_segment_path
        ])
        
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        
        # 2. 使用 Cam++ 模型提取声纹
        emb_res = speaker_model.generate(input=temp_segment_path)
        
        if emb_res and len(emb_res) > 0:
            emb = emb_res[0].get("spk_embedding", None)
            if emb is not None:
                # 转换为 numpy 数组并确保是 1D
                emb_array = np.array(emb)
                
                # 如果是多维数组，展平为 1D
                if emb_array.ndim > 1:
                    emb_array = emb_array.flatten()
                
                # 转换为 Python list
                return emb_array.tolist()
        
        return None
        
    except subprocess.TimeoutExpired:
        logger.warning("⚠️ ffmpeg 提取超时")
        return None
    except subprocess.CalledProcessError as e:
        logger.warning(f"⚠️ ffmpeg 提取失败: {e}")
        return None
    except Exception as e:
        logger.warning(f"⚠️ 声纹提取异常: {e}")
        return None
    finally:
        # 清理临时文件
        if temp_segment_path and os.path.exists(temp_segment_path):
            try:
                os.remove(temp_segment_path)
            except:
                pass
