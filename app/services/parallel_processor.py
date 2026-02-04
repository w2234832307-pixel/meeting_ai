"""
并行处理模块：实现 FunASR 和 Pyannote 的并行处理
按照新流程：
1. 并行执行 FunASR (字级别时间戳) 和 Pyannote (RTTM)
2. 字级别映射说话人
3. 按说话人聚合句子
4. 声纹匹配
"""
import logging
from typing import List, Dict, Any, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logger = logging.getLogger(__name__)


def map_words_to_speakers(
    words: List[Dict[str, Any]], 
    rttm_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    将字级别时间戳映射到说话人
    
    Args:
        words: 字级别识别结果，格式: [{"char": "你", "start": 0.5, "end": 0.6}, ...]
        rttm_segments: Pyannote RTTM 格式，格式: [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}, ...]
    
    Returns:
        带说话人ID的字列表
    """
    mapped_words = []
    
    # 对 RTTM 片段按时间排序
    sorted_rttm = sorted(rttm_segments, key=lambda x: x['start'])
    
    for word in words:
        char = word.get('char', '')
        start = word.get('start', 0)
        end = word.get('end', 0)
        
        # 计算字的中心时间点
        center_time = (start + end) / 2.0
        
        # 找到包含中心时间点的说话人片段
        speaker_id = None
        for seg in sorted_rttm:
            if seg['start'] <= center_time <= seg['end']:
                speaker_id = seg['speaker']
                break
        
        # 如果没找到，找最近的片段
        if speaker_id is None:
            min_distance = float('inf')
            for seg in sorted_rttm:
                # 计算到片段中心的距离
                seg_center = (seg['start'] + seg['end']) / 2.0
                distance = abs(center_time - seg_center)
                if distance < min_distance:
                    min_distance = distance
                    speaker_id = seg['speaker']
        
        mapped_words.append({
            'char': char,
            'start': start,
            'end': end,
            'speaker_id': speaker_id or 'SPEAKER_00'
        })
    
    return mapped_words


def aggregate_by_speaker(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    按说话人聚合句子
    
    Args:
        words: 带说话人ID的字列表
    
    Returns:
        聚合后的句子列表，格式: [{"text": "你好", "start": 0.5, "end": 1.2, "speaker_id": "SPEAKER_00"}, ...]
    """
    if not words:
        return []
    
    sentences = []
    current_sentence = []
    current_speaker = None
    sentence_start = None
    
    for word in words:
        char = word.get('char', '')
        speaker_id = word.get('speaker_id')
        start = word.get('start', 0)
        end = word.get('end', 0)
        
        # 如果说话人改变，或者遇到标点符号，结束当前句子
        if (current_speaker is not None and speaker_id != current_speaker) or char in ['。', '！', '？', '.', '!', '?', '\n']:
            if current_sentence:
                sentences.append({
                    'text': ''.join(current_sentence),
                    'start': sentence_start,
                    'end': word.get('start', 0),
                    'speaker_id': current_speaker
                })
                current_sentence = []
                current_speaker = None
                sentence_start = None
        
        # 开始新句子或继续当前句子
        if current_speaker is None:
            current_speaker = speaker_id
            sentence_start = start
        
        current_sentence.append(char)
    
    # 处理最后一个句子
    if current_sentence and current_speaker is not None:
        sentences.append({
            'text': ''.join(current_sentence),
            'start': sentence_start,
            'end': words[-1].get('end', 0),
            'speaker_id': current_speaker
        })
    
    return sentences


def parse_rttm(rttm_content: str) -> List[Dict[str, Any]]:
    """
    解析 RTTM 格式文件
    
    RTTM 格式：
    SPEAKER <file> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>
    
    Args:
        rttm_content: RTTM 文件内容
    
    Returns:
        说话人片段列表: [{"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"}, ...]
    """
    segments = []
    for line in rttm_content.strip().split('\n'):
        if not line.strip() or line.startswith(';;'):
            continue
        
        parts = line.strip().split()
        if len(parts) >= 8 and parts[0] == 'SPEAKER':
            start = float(parts[3])
            duration = float(parts[4])
            speaker = parts[7]
            
            segments.append({
                'start': start,
                'end': start + duration,
                'speaker': speaker
            })
    
    return sorted(segments, key=lambda x: x['start'])
