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
from typing import List, Dict
import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)


def perform_speaker_diarization_with_vad(
    audio_path: str,
    vad_segments: List,
    speaker_model,
    device: str = "cuda",
    min_segment_duration: float = 1.0,
    distance_threshold: float = 0.5
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
        
        # 3. 使用层次聚类进行说话人分离
        logger.info(f"🔬 进行说话人聚类...")
        
        clustering = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=distance_threshold,  # 距离阈值
            metric='cosine',
            linkage='average'
        )
        
        embeddings_array = np.array(embeddings)
        cluster_labels = clustering.fit_predict(embeddings_array)
        
        # 4. 将聚类结果映射到片段
        for idx, segment in enumerate(valid_segments):
            segment["speaker_id"] = str(cluster_labels[idx])
        
        n_speakers = len(set(cluster_labels))
        logger.info(f"✅ 识别出 {n_speakers} 个说话人")
        
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
                if hasattr(emb, 'tolist'):
                    emb = emb.tolist()
                return emb
        
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
