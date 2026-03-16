"""
并行处理模块：实现 FunASR 和 Pyannote 的并行处理
按照新流程：
1. 并行执行 FunASR (字级别时间戳) 和 Pyannote (RTTM)
2. 字级别映射说话人 (基于最大重叠与边界吸附)
3. 按说话人聚合句子 (修复标点断句逻辑)
4. 声纹匹配
"""
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

def map_words_to_speakers(
    words: List[Dict[str, Any]], 
    rttm_segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """将字级别时间戳精确映射到说话人（使用最大重叠 + 边界吸附）"""
    mapped_words = []
    sorted_rttm = sorted(rttm_segments, key=lambda x: x['start'])
    
    for word in words:
        char = word.get('char', '')
        start = word.get('start', 0.0)
        end = word.get('end', 0.0)
        
        best_speaker = None
        max_overlap = 0.0
        
        # 1. 优先寻找重叠最多的片段 (解决字跨越边界的问题)
        for seg in sorted_rttm:
            overlap = max(0.0, min(end, seg['end']) - max(start, seg['start']))
            if overlap > max_overlap:
                max_overlap = overlap
                best_speaker = seg['speaker']
        
        # 2. 如果完全没有重叠（掉在静音缝隙里），寻找距离"边界"最近的片段
        if best_speaker is None:
            min_dist = float('inf')
            for seg in sorted_rttm:
                # 计算字到该片段的最短距离（前置距离或后置距离）
                dist = max(0.0, max(seg['start'] - end, start - seg['end']))
                if dist < min_dist:
                    min_dist = dist
                    best_speaker = seg['speaker']
        
        mapped_words.append({
            'char': char,
            'start': start,
            'end': end,
            'speaker_id': best_speaker or 'SPEAKER_00'
        })
    
    return mapped_words


def aggregate_by_speaker(words: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """按说话人聚合句子（完美解决标点符号遗留问题）"""
    if not words:
        return []
    
    sentences = []
    current_sentence = []
    current_speaker = None
    sentence_start = None
    last_end = 0.0
    
    for word in words:
        char = word.get('char', '')
        speaker_id = word.get('speaker_id')
        start = word.get('start', 0.0)
        end = word.get('end', 0.0)
        
        # 1. 发现说话人换了，先结算上一句（不含当前字）
        if current_speaker is not None and speaker_id != current_speaker:
            if current_sentence:
                sentences.append({
                    'text': ''.join(current_sentence),
                    'start': sentence_start,
                    'end': last_end,
                    'speaker_id': current_speaker
                })
            current_sentence = []
            current_speaker = None
            sentence_start = None
            
        # 2. 初始化新句子的属性
        if current_speaker is None:
            current_speaker = speaker_id
            sentence_start = start
            
        # 3. 先把字/标点加进当前的句子里！
        current_sentence.append(char)
        last_end = end
        
        # 4. 如果这个字恰好是个标点符号，连带标点一起结算成一整句！
        if char in ['。', '！', '？', '.', '!', '?', '\n']:
            sentences.append({
                'text': ''.join(current_sentence),
                'start': sentence_start,
                'end': last_end,
                'speaker_id': current_speaker
            })
            # 结算完清空，准备迎接下一句
            current_sentence = []
            current_speaker = None
            sentence_start = None

    # 处理收尾（如果最后一句没有标点符号）
    if current_sentence and current_speaker is not None:
        sentences.append({
            'text': ''.join(current_sentence),
            'start': sentence_start,
            'end': last_end,
            'speaker_id': current_speaker
        })
    
    return sentences


def parse_rttm(rttm_content: str) -> List[Dict[str, Any]]:
    """解析 RTTM 格式文件 (保持不变)"""
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