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


def aggregate_by_speaker(words: List[Dict[str, Any]], min_sentence_length: int = 3, max_pause: float = 1.5) -> List[Dict[str, Any]]:
    """
    按说话人聚合句子（优化版：减少片段数量）
    
    Args:
        words: 带说话人ID的字列表
        min_sentence_length: 最小句子长度（字符数），短于此长度的片段会被合并
        max_pause: 最大停顿时间（秒），超过此时间的停顿会切分句子
    
    Returns:
        聚合后的句子列表，格式: [{"text": "你好", "start": 0.5, "end": 1.2, "speaker_id": "SPEAKER_00"}, ...]
    """
    if not words:
        return []
    
    sentences = []
    current_sentence = []
    current_speaker = None
    sentence_start = None
    last_word_end = None
    
    for i, word in enumerate(words):
        char = word.get('char', '')
        speaker_id = word.get('speaker_id')
        start = word.get('start', 0)
        end = word.get('end', 0)
        
        # 计算与上一个字的时间间隔
        time_gap = start - last_word_end if last_word_end is not None else 0
        
        # 判断是否需要切分：
        # 1. 说话人改变且时间间隔较大（>0.3秒，避免误识别导致的频繁切换）
        # 2. 遇到句号类标点且时间间隔较大（表示真正的句子结束）
        # 3. 时间间隔过大（表示长时间停顿）
        speaker_changed = current_speaker is not None and speaker_id != current_speaker
        is_sentence_end = char in ['。', '！', '？', '.', '!', '?']
        long_pause = time_gap > max_pause
        
        should_split = False
        if speaker_changed and time_gap > 0.3:  # 说话人切换且间隔>0.3秒
            should_split = True
        elif is_sentence_end and time_gap > 0.5:  # 句号且间隔>0.5秒
            should_split = True
        elif long_pause:  # 长时间停顿
            should_split = True
        
        if should_split and current_sentence:
            # 结束当前句子
            sentences.append({
                'text': ''.join(current_sentence),
                'start': sentence_start,
                'end': last_word_end or start,
                'speaker_id': current_speaker
            })
            current_sentence = []
            current_speaker = None
            sentence_start = None
        
        # 开始新句子或继续当前句子
        if current_speaker is None:
            current_speaker = speaker_id
            sentence_start = start
        
        # 添加字符（包括标点符号，它们应该属于句子的一部分）
        current_sentence.append(char)
        last_word_end = end
    
    # 处理最后一个句子
    if current_sentence and current_speaker is not None:
        sentences.append({
            'text': ''.join(current_sentence),
            'start': sentence_start,
            'end': words[-1].get('end', 0),
            'speaker_id': current_speaker
        })
    
    # 后处理：合并相邻的相同说话人的短片段
    if len(sentences) <= 1:
        return sentences
    
    merged_sentences = []
    for i, sent in enumerate(sentences):
        if not merged_sentences:
            merged_sentences.append(sent)
            continue
        
        prev_sent = merged_sentences[-1]
        
        # 如果前一个片段很短，且说话人相同，且时间间隔不大，则合并
        should_merge = (
            prev_sent['speaker_id'] == sent['speaker_id'] and
            len(prev_sent['text'].strip()) < min_sentence_length and
            sent['start'] - prev_sent['end'] < 0.5  # 时间间隔小于0.5秒
        )
        
        if should_merge:
            # 合并到前一个片段
            merged_sentences[-1] = {
                'text': prev_sent['text'] + sent['text'],
                'start': prev_sent['start'],
                'end': sent['end'],
                'speaker_id': prev_sent['speaker_id']
            }
        else:
            merged_sentences.append(sent)
    
    return merged_sentences


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
            try:
                start = float(parts[3])
                duration = float(parts[4])
            except ValueError:
                continue

            # 兼容各种 RTTM 变体：优先从整行中寻找形如 SPEAKER_XX 的标签
            speaker = None
            for p in reversed(parts):
                if p.startswith("SPEAKER_"):
                    speaker = p
                    break

            # 如果找不到标准标签，退回到第 8 列；再不行就给个默认值
            if speaker is None:
                raw = parts[7] if len(parts) > 7 else "<NA>"
                speaker = raw if raw != "<NA>" else "SPEAKER_00"

            segments.append({
                "start": start,
                "end": start + duration,
                "speaker": speaker,
            })

    return sorted(segments, key=lambda x: x["start"])
