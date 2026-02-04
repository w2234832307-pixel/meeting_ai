"""
字级别 ASR 输出模块
从 FunASR SenseVoiceSmall 结果中提取字级别时间戳
"""
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def extract_word_level_timestamps(asr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 ASR 结果中提取字级别时间戳
    
    Args:
        asr_result: ASR 识别结果，可能包含 timestamp 或 sentences 字段
    
    Returns:
        字级别时间戳列表: [{"char": "你", "start": 0.5, "end": 0.6}, ...]
    """
    words = []
    
    # 检查是否有 timestamp 字段（字级别）
    timestamp = asr_result.get("timestamp", [])
    if timestamp and len(timestamp) > 0:
        for ts_item in timestamp:
            if isinstance(ts_item, list) and len(ts_item) >= 2:
                start = ts_item[0] / 1000.0 if isinstance(ts_item[0], (int, float)) else 0.0
                end = ts_item[1] / 1000.0 if isinstance(ts_item[1], (int, float)) else start + 0.1
                char = ts_item[-1] if len(ts_item) > 2 else ""
                
                if char:
                    words.append({
                        "char": char,
                        "start": round(start, 3),
                        "end": round(end, 3)
                    })
    
    # 如果没有字级别时间戳，尝试从 sentences 中提取
    if not words:
        sentences = asr_result.get("sentences", [])
        if sentences:
            for sent in sentences:
                sent_text = sent.get("text", "")
                sent_timestamp = sent.get("timestamp", [])
                
                if sent_timestamp and len(sent_timestamp) >= 2:
                    sent_start = sent_timestamp[0] / 1000.0 if isinstance(sent_timestamp[0], (int, float)) else 0.0
                    sent_end = sent_timestamp[1] / 1000.0 if isinstance(sent_timestamp[1], (int, float)) else sent_start + 1.0
                    
                    # 按字符平均分配时间
                    if sent_text:
                        char_duration = (sent_end - sent_start) / len(sent_text)
                        for i, char in enumerate(sent_text):
                            words.append({
                                "char": char,
                                "start": round(sent_start + i * char_duration, 3),
                                "end": round(sent_start + (i + 1) * char_duration, 3)
                            })
    
    # 如果还是没有，从 text 中按平均时间分配
    if not words:
        text = asr_result.get("text", "")
        start_time = asr_result.get("start_time", 0.0)
        end_time = asr_result.get("end_time", 0.0)
        
        if text and end_time > start_time:
            char_duration = (end_time - start_time) / len(text)
            for i, char in enumerate(text):
                words.append({
                    "char": char,
                    "start": round(start_time + i * char_duration, 3),
                    "end": round(start_time + (i + 1) * char_duration, 3)
                })
    
    return words
