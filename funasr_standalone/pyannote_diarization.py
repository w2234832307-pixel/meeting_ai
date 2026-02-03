#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pyannote 说话人分离模块
使用专业的 Pyannote.audio 模型进行说话人分离
"""
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False
    logger.warning("⚠️ Pyannote.audio 未安装，说话人分离功能将不可用")
    logger.warning("   安装命令: pip install pyannote.audio")


def perform_pyannote_diarization(
    audio_path: str,
    transcript: List[Dict],
    use_auth_token: Optional[str] = None
) -> List[Dict]:
    """
    使用 Pyannote 进行说话人分离
    
    Args:
        audio_path: 音频文件路径
        transcript: ASR识别结果，包含text、start_time、end_time
        use_auth_token: HuggingFace token（如果需要访问私有模型）
    
    Returns:
        更新后的transcript，包含speaker_id字段
    """
    if not PYANNOTE_AVAILABLE:
        logger.error("❌ Pyannote.audio 未安装，无法使用说话人分离")
        logger.error("   请运行: pip install pyannote.audio")
        # 返回原始transcript，所有片段标记为speaker_id="0"
        for item in transcript:
            if 'speaker_id' not in item:
                item['speaker_id'] = "0"
        return transcript
    
    try:
        logger.info("🎤 使用 Pyannote.audio 进行说话人分离...")
        
        # 加载预训练模型
        # 注意：首次使用需要HuggingFace token，并在HuggingFace上接受模型使用协议
        # https://huggingface.co/pyannote/speaker-diarization-3.1
        try:
            import os
            # 优先使用传入的 token，其次从环境变量读取
            hf_token = use_auth_token or os.getenv("HF_TOKEN")
            if hf_token:
                # 新版本的 transformers 使用 token 参数，而不是 use_auth_token
                try:
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        token=hf_token
                    )
                except TypeError:
                    # 兼容旧版本，如果 token 参数不支持，尝试 use_auth_token
                    pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=hf_token
                    )
            else:
                # 尝试不使用token（如果模型是公开的）
                pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
        except Exception as e:
            logger.error(f"❌ 加载 Pyannote 模型失败: {e}")
            logger.error("   请确保:")
            logger.error("   1. 已安装 pyannote.audio: pip install pyannote.audio")
            logger.error("   2. 在 HuggingFace 上接受模型使用协议: https://huggingface.co/pyannote/speaker-diarization-3.1")
            logger.error("   3. 如果需要，提供 token 参数（或通过 HF_TOKEN 环境变量）")
            # 降级：返回原始transcript
            for item in transcript:
                if 'speaker_id' not in item:
                    item['speaker_id'] = "0"
            return transcript
        
        # 处理音频
        logger.info(f"📂 处理音频文件: {audio_path}")
        diarization = pipeline(audio_path)
        
        # 构建说话人时间映射
        # diarization格式: (start, end, speaker_label)
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                'start_time': turn.start,
                'end_time': turn.end,
                'speaker_id': speaker
            })
        
        logger.info(f"✅ Pyannote 识别出 {len(set(s['speaker_id'] for s in speaker_segments))} 个说话人")
        logger.info(f"   共 {len(speaker_segments)} 个说话人片段")
        
        # 将说话人信息映射到transcript
        # 对于每个transcript片段，找到时间重叠的说话人片段
        for item in transcript:
            item_start = item.get('start_time', 0)
            item_end = item.get('end_time', 0)
            
            # 找到时间重叠的说话人片段
            matched_speaker = None
            max_overlap = 0
            
            for seg in speaker_segments:
                seg_start = seg['start_time']
                seg_end = seg['end_time']
                
                # 计算重叠时间
                overlap_start = max(item_start, seg_start)
                overlap_end = min(item_end, seg_end)
                overlap = max(0, overlap_end - overlap_start)
                
                # 如果重叠时间超过片段长度的50%，认为是匹配的
                item_duration = item_end - item_start
                if item_duration > 0 and overlap / item_duration > 0.5:
                    if overlap > max_overlap:
                        max_overlap = overlap
                        matched_speaker = seg['speaker_id']
            
            # 如果找到匹配的说话人，使用它；否则使用最近的说话人
            if matched_speaker:
                item['speaker_id'] = matched_speaker
            else:
                # 找到最近的说话人片段
                min_distance = float('inf')
                nearest_speaker = None
                
                for seg in speaker_segments:
                    seg_start = seg['start_time']
                    seg_end = seg['end_time']
                    seg_center = (seg_start + seg_end) / 2
                    item_center = (item_start + item_end) / 2
                    
                    distance = abs(item_center - seg_center)
                    if distance < min_distance:
                        min_distance = distance
                        nearest_speaker = seg['speaker_id']
                
                item['speaker_id'] = nearest_speaker if nearest_speaker else "SPEAKER_00"
        
        # 规范化说话人ID（从SPEAKER_00, SPEAKER_01... 转换为 0, 1, 2...）
        speaker_id_map = {}
        speaker_counter = 0
        
        for item in transcript:
            original_id = item.get('speaker_id', 'SPEAKER_00')
            if original_id not in speaker_id_map:
                speaker_id_map[original_id] = str(speaker_counter)
                speaker_counter += 1
            item['speaker_id'] = speaker_id_map[original_id]
        
        logger.info(f"✅ 说话人分离完成，共识别出 {len(speaker_id_map)} 个说话人")
        
        return transcript
        
    except Exception as e:
        logger.error(f"❌ Pyannote 说话人分离失败: {e}", exc_info=True)
        # 降级：返回原始transcript，所有片段标记为speaker_id="0"
        for item in transcript:
            if 'speaker_id' not in item:
                item['speaker_id'] = "0"
        return transcript
