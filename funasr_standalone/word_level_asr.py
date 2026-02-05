"""
字级别 ASR 输出模块
从 FunASR SenseVoiceSmall 结果中提取字级别时间戳
"""
import logging
import re
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


def clean_funasr_char(char: str) -> str:
    """
    清理单个字符中的 FunASR 特殊标记
    如果字符是标记的一部分，返回空字符串
    """
    if not char:
        return char
    
    # 如果字符是标记的一部分（包含 <、>、| 等），直接过滤掉
    if char in ['<', '>', '|']:
        return ''
    
    # 如果字符是标记中的一部分（如 'z', 'h' 在 '<|zh|>' 中），需要上下文判断
    # 但这里我们只处理单个字符，所以简单过滤掉特殊字符
    if char.strip() == '' and char not in [' ', '\t', '\n']:
        return ''
    
    return char


def clean_funasr_text(text: str) -> str:
    """
    清理 FunASR 文本中的特殊标记
    """
    if not text:
        return text
    
    # 移除所有包含特殊字符的标记
    # 1. 移除 <|...|> 格式
    text = re.sub(r'<\|[^>]*\|>', '', text)
    # 2. 移除包含竖线的标记 <...|...>
    text = re.sub(r'<[^>]*\|[^>]*>', '', text)
    # 3. 移除包含 FunASR 关键词的标记
    funasr_keywords = ['EMO', 'NEUTRAL', 'Speech', 'zh', 'withitn', 'UNKNOWN', 'EMOTION']
    for keyword in funasr_keywords:
        text = re.sub(rf'<[^>]*{keyword}[^>]*>', '', text, flags=re.IGNORECASE)
    # 4. 移除全大写字母的标记 <UPPERCASE>
    text = re.sub(r'<[A-Z][A-Z0-9_]{2,}>', '', text)
    # 5. 清理多余空格
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def extract_word_level_timestamps(asr_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    从 ASR 结果中提取字级别时间戳
    
    Args:
        asr_result: ASR 识别结果，可能包含 timestamp 或 sentences 字段
    
    Returns:
        字级别时间戳列表: [{"char": "你", "start": 0.5, "end": 0.6}, ...]
    """
    words = []
    
    # 调试：打印ASR结果结构
    if not words:
        logger.debug(f"🔍 提取字级别时间戳，ASR结果键: {list(asr_result.keys())}")
    
    # 检查是否有 timestamp 字段（字级别）
    timestamp = asr_result.get("timestamp", [])
    if timestamp and len(timestamp) > 0:
        logger.debug(f"🔍 使用timestamp字段，共 {len(timestamp)} 个时间戳")
        for ts_item in timestamp:
            if isinstance(ts_item, list) and len(ts_item) >= 2:
                start = ts_item[0] / 1000.0 if isinstance(ts_item[0], (int, float)) else 0.0
                end = ts_item[1] / 1000.0 if isinstance(ts_item[1], (int, float)) else start + 0.1
                char = ts_item[-1] if len(ts_item) > 2 else ""
                
                # 清理字符中的标记
                char = clean_funasr_char(char)
                
                # 只添加非空字符
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
            logger.debug(f"🔍 使用sentences字段，共 {len(sentences)} 个句子")
            for sent in sentences:
                sent_text = sent.get("text", "")
                # 先清理整个文本
                sent_text = clean_funasr_text(sent_text)
                
                sent_timestamp = sent.get("timestamp", [])
                
                if sent_timestamp and len(sent_timestamp) >= 2:
                    sent_start = sent_timestamp[0] / 1000.0 if isinstance(sent_timestamp[0], (int, float)) else 0.0
                    sent_end = sent_timestamp[1] / 1000.0 if isinstance(sent_timestamp[1], (int, float)) else sent_start + 1.0
                    
                    # 按字符平均分配时间
                    if sent_text:
                        char_duration = (sent_end - sent_start) / len(sent_text)
                        for i, char in enumerate(sent_text):
                            # 再次清理单个字符
                            char = clean_funasr_char(char)
                            if char:  # 只添加非空字符
                                words.append({
                                    "char": char,
                                    "start": round(sent_start + i * char_duration, 3),
                                    "end": round(sent_start + (i + 1) * char_duration, 3)
                                })
    
    # 如果还是没有，从 text 中按平均时间分配
    if not words:
        text = asr_result.get("text", "")
        # 先清理整个文本
        text = clean_funasr_text(text)
        
        start_time = asr_result.get("start_time", 0.0)
        end_time = asr_result.get("end_time", 0.0)
        
        if text:
            logger.debug(f"🔍 使用text字段，文本长度: {len(text)}")
            # 如果没有时间信息，使用默认值
            if end_time <= start_time:
                end_time = start_time + len(text) * 0.1  # 假设每个字0.1秒
        
        if text and end_time > start_time:
            char_duration = (end_time - start_time) / len(text)
            for i, char in enumerate(text):
                # 再次清理单个字符
                char = clean_funasr_char(char)
                if char:  # 只添加非空字符
                    words.append({
                        "char": char,
                        "start": round(start_time + i * char_duration, 3),
                        "end": round(start_time + (i + 1) * char_duration, 3)
                    })
    
    return words
