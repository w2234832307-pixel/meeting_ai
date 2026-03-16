import os, shutil, uuid, tempfile, markdown, requests, traceback, json
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import asyncio
from app.core.config import settings
from app.core.logger import logger
from app.schemas.task import MeetingResponse, ArchiveRequest, ArchiveResponse, TranscriptItem, SpeakerSummary
from app.services.vector import vector_service
from app.services.asr_factory import get_asr_service_by_name
from app.services.llm_factory import get_llm_service_by_name
from app.services.document import document_service 
from app.services.prompt_template import prompt_template_service
import time
import httpx
import glob
import re
import traceback



router = APIRouter()

# --- 辅助工具函数 ---



def cleanup_files(files: list):
    """请求结束时的即时清理（允许失败）"""
    for f in files:
        if f and os.path.exists(f):
            try:
                os.remove(f)
                logger.debug(f"🧹 已清理临时文件: {f}")
            except Exception as e:
                # 降级为 debug，因为文件锁定失败是高并发下的正常现象
                logger.debug(f"⚠️ 临时文件占用中，留待全局回收: {f}")

def global_temp_sweep(temp_dir: str, max_age_hours: int = 2):
    """
    全局兜底清扫：清理存活时间超过指定小时数的废弃文件
    建议在 main.py 的 startup 事件中调用
    """
    try:
        now = time.time()
        # 假设你的临时文件都有特定的前缀，比如 url_dl_, doc_, multi_, speaker_
        search_pattern = os.path.join(temp_dir, "*")
        
        cleaned_count = 0
        for file_path in glob.glob(search_pattern):
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                # 如果文件最后修改时间距离现在超过了 max_age_hours
                if (now - file_mtime) > (max_age_hours * 3600):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except:
                        pass
        if cleaned_count > 0:
            logger.info(f"♻️ 全局垃圾回收完成，清理了 {cleaned_count} 个滞留临时文件")
    except Exception as e:
        logger.error(f"❌ 全局临时文件清扫失败: {e}")

def get_audio_duration(audio_path: str) -> float:
    """
    获取音频文件时长（秒）
    
    Args:
        audio_path: 音频文件路径
        
    Returns:
        音频时长（秒），失败返回0.0
    """
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        duration = info.frames / info.samplerate
        return duration
    except Exception:
        # 如果soundfile不支持，尝试用ffprobe
        try:
            import subprocess
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', audio_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception:
            pass
    return 0.0

def make_transcript_timestamps_continuous(
    segments: List[Dict[str, Any]],
) -> None:
    """
    将 transcript 列表中的时间戳调整为连续时间轴（毫秒）：
    - 如果发现下一个片段的 start_time > 当前片段的 end_time，
      则将当前片段的 end_time 补齐为下一个片段的 start_time。
    - 同时确保每个片段 end_time >= start_time。
    原地修改，不返回值。
    """
    if not segments:
        return

    # 按开始时间排序，避免乱序导致的时间轴问题
    segments.sort(key=lambda x: x.get("start_time", 0))

    prev = segments[0]
    # 确保第一个片段合法（毫秒）
    if prev.get("end_time", 0) <= prev.get("start_time", 0):
        prev["end_time"] = int(prev.get("start_time", 0)) + 10  # 至少10毫秒

    for cur in segments[1:]:
        cur_start = int(cur.get("start_time", 0))
        prev_end = int(prev.get("end_time", prev.get("start_time", 0)))

        # 保证当前片段至少有一个极小长度（毫秒）
        if int(cur.get("end_time", cur_start)) <= cur_start:
            cur["end_time"] = cur_start + 10  # 至少10毫秒

        # 如果中间有空白，把空白并到前一段的尾部
        if cur_start > prev_end:
            prev["end_time"] = cur_start

        prev = cur

def build_word_level_from_transcript(
    segments: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """
    基于 transcript 段落，按字符平均分配时间，构造字级别的 words：
    - 每个字符一个 word：{"char": "你", "start": x, "end": y, "speaker_id": ..., "audio_id": ...}
    - 适用于 funasr 和第三方 ASR（只要有 transcript 即可）
    - 注意：segments 中的 start_time/end_time 是毫秒，但 words 中的 start/end 是秒（用于后续计算偏移）
    """
    words: List[Dict[str, Any]] = []
    if not segments:
        return words

    # 先按时间排序，保证时间线一致
    sorted_segs = sorted(segments, key=lambda x: x.get("start_time", 0))

    for seg in sorted_segs:
        text = (seg.get("text") or "").strip()
        if not text:
            continue

        # 从毫秒转换为秒（用于后续计算）
        start_ms = int(seg.get("start_time", 0))
        end_ms = int(seg.get("end_time", start_ms))
        if end_ms <= start_ms:
            end_ms = start_ms + max(len(text) * 50, 100)  # 至少100毫秒或每字符50毫秒

        start_sec = start_ms / 1000.0
        end_sec = end_ms / 1000.0
        duration = end_sec - start_sec
        char_count = len(text)
        if char_count <= 0 or duration <= 0:
            continue

        char_duration_sec = duration / char_count
        char_duration_ms = (end_ms - start_ms) / max(char_count, 1)
        speaker_id = seg.get("speaker_id")
        audio_id = seg.get("audio_id")

        seg_words: List[Dict[str, Any]] = []

        for i, ch in enumerate(text):
            # 绝对时间（秒）- 保持原有返回值，供需要时使用
            ch_start = start_sec + i * char_duration_sec
            ch_end = start_sec + (i + 1) * char_duration_sec
            words.append(
                {
                    "char": ch,
                    "start": round(ch_start, 3),
                    "end": round(ch_end, 3),
                    "speaker_id": speaker_id,
                    "audio_id": audio_id,
                }
            )

            # 相对当前段起点的偏移（毫秒），直接按字符平均分配，保证：
            # - 单字符片段不交叠
            # - 不会跨出当前段的 start_time/end_time
            offset_start_ms = int(i * char_duration_ms)
            offset_end_ms = int((i + 1) * char_duration_ms)
            seg_words.append(
                {
                    "offsetStartMs": offset_start_ms,
                    "offsetEndMs": offset_end_ms,
                    "word": ch,
                }
            )

        if seg_words:
            seg["words"] = seg_words

    return words

def attach_words_to_transcript_segments(
    transcript: List[Dict[str, Any]],
    words: List[Dict[str, Any]],
) -> None:
    """
    将字级别 words 以类似三方返回的格式挂到每段 transcript 下：
    每个 transcript 段新增一个字段：
      "words": [
        {"offsetStartMs": 0, "offsetEndMs": 120, "word": "特曼"},
        ...
      ]
    其中 offsetStartMs/offsetEndMs 是相对于该段 start_time 的偏移（毫秒）。
    现阶段我们按字符级别构造，word 字段为单个字符。
    """
    if not transcript or not words:
        return

    # 按开始时间排序，确保时间线一致
    transcript.sort(key=lambda x: x.get("start_time", 0.0))
    words_sorted = sorted(words, key=lambda w: w.get("start", 0.0))

    # 为了效率，使用指针遍历 words
    w_idx = 0
    w_len = len(words_sorted)

    for seg in transcript:
        # transcript 中的时间是毫秒
        seg_start_ms = int(seg.get("start_time", 0))
        seg_end_ms = int(seg.get("end_time", seg_start_ms))
        seg_start_sec = seg_start_ms / 1000.0
        seg_end_sec = seg_end_ms / 1000.0
        seg_words: List[Dict[str, Any]] = []

        # 从当前指针位置开始，收集属于这个段的 words
        while w_idx < w_len:
            w = words_sorted[w_idx]
            ws = float(w.get("start", 0.0))
            we = float(w.get("end", ws))

            # 如果这个字在当前段之前，移动指针
            if we <= seg_start_sec:
                w_idx += 1
                continue

            # 如果这个字开始时间已经超过本段结束，则该段收集结束
            if ws >= seg_end_sec:
                break

            # 落在当前段时间范围内的字（计算相对于段开始的偏移，毫秒）
            offset_start_ms = int(max(ws - seg_start_sec, 0.0) * 1000)
            offset_end_ms = int(max(we - seg_start_sec, 0.0) * 1000)
            seg_words.append(
                {
                    "offsetStartMs": offset_start_ms,
                    "offsetEndMs": offset_end_ms,
                    "word": w.get("char", ""),
                }
            )
            w_idx += 1

        # 只在有内容时挂 words 字段，避免污染原有结构
        if seg_words:
            seg["words"] = seg_words

def format_transcript_for_llm(transcript_data: List[Dict]) -> str:
    """
    将带说话人信息的 transcript_data 格式化为易读的文本格式。
    为了最大限度减少 token 消耗，这里完全不输出音频名称和时间戳，只保留「谁说了什么」。
    
    Args:
        transcript_data: 转录数据列表，包含 text, start_time, end_time, speaker_id, audio_name
        
    Returns:
        格式化后的文本字符串，格式：
            说话人A: 文本
            说话人B: 文本
            ...
    """
    if not transcript_data:
        return ""
    
    # 全局按开始时间排序，忽略 audio_name 和时间戳输出，只保留“谁说了什么”
    items_sorted = sorted(transcript_data, key=lambda x: x.get("start_time", 0.0))
    
    lines: List[str] = []
    for item in items_sorted:
        text = item.get("text", "").strip()
        if not text:
            continue
        
        speaker_id = item.get("speaker_id")
        if speaker_id is None:
            speaker_str = "未知"
        elif isinstance(speaker_id, (int, str)) and str(speaker_id).isdigit():
            speaker_str = f"说话人{speaker_id}"
        else:
            speaker_str = str(speaker_id)
        
        lines.append(f"{speaker_str}: {text}")
    
    return "\n".join(lines)


async def is_modify_minutes_intent(user_requirement: Optional[str]) -> bool:
    """
    使用专门的意图 LLM（通过 INTENT_LLM_* 配置）判断用户是否意图“在现有会议纪要基础上修改/润色”。
    优先走这套独立的 LLM 服务，只有在调用失败时才退回到极简关键词兜底。
    """
    if not user_requirement or not user_requirement.strip():
        return False

    # 从 .env 中读取专用意图模型配置
    base_url = os.getenv("INTENT_LLM_BASE_URL") or os.getenv("LLM_BASE_URL")
    api_key = os.getenv("INTENT_LLM_API_KEY") or os.getenv("LLM_API_KEY")
    model_name = os.getenv("INTENT_LLM_MODEL_NAME") or os.getenv("LLM_MODEL_NAME")

    if not base_url or not api_key or not model_name:
        logger.warning("⚠️ 意图识别 LLM 配置不完整，回退到关键词兜底判断")
    else:
        try:
            # 延迟导入，避免在未使用时引入额外依赖
            from openai import OpenAI

            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

            prompt = f"""
你是一个“用户意图分类器”，只负责判断下面这段用户要求是不是在说：
“请在现有会议纪要的基础上进行修改/润色/微调，而不是重新从头生成新的会议纪要。”

请只根据【用户要求】本身来判断，不要考虑其它上下文。

【输出要求】：
1. 如果用户主要意图是“基于现有纪要修改/润色/继续完善”，请只输出：YES
2. 如果用户主要意图是“重新生成新的纪要 / 不依赖现有纪要”，请只输出：NO
3. 不要输出任何解释，不要输出其它文字。

【用户要求】：
{user_requirement}
"""

            # 同步客户端放到线程池中执行，避免阻塞事件循环
            def _call_intent_llm() -> str:
                resp = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=8,
                )
                return (resp.choices[0].message.content or "").strip()

            response_text = await asyncio.to_thread(_call_intent_llm)
            answer = (response_text or "").strip().lower()

            # 严格按 YES / NO 判定，兼容中英混合前缀
            if answer.startswith("yes") or answer.startswith("是"):
                return True
            if answer.startswith("no") or answer.startswith("否"):
                return False

            logger.warning(f"⚠️ 意图识别LLM返回无法直接判定的结果: {response_text}")

        except Exception as e:
            logger.warning(f"⚠️ 意图识别LLM调用失败，回退到关键词兜底: {e}")

    # 兜底：极简的关键词匹配（只在 LLM 不可用或返回异常结果时使用）
    text = user_requirement.strip().lower()
    fallback_keywords = [
        "在现有会议纪要基础上", "在原有会议纪要基础上", "在原纪要基础上",
        "在这个纪要基础上", "在上述纪要基础上", "在上面的纪要基础上",
        "在当前纪要基础上", "基于当前纪要", "基于这个纪要", "基于上述纪要",
        "修改会议纪要", "调整会议纪要", "润色会议纪要", "优化会议纪要",
        "微调纪要", "不要重头生成", "不要重新生成", "不要重新写",
        "保留原有结构", "保留原来的纪要", "对上面的纪要做修改", "对上述纪要做修改",
    ]
    for kw in fallback_keywords:
        if kw.lower() in text:
            return True
    return False


# logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# 1. 说话人摘要生成模块
# -----------------------------------------------------------------------------
async def generate_speaker_summaries(
    transcript_data: List[Dict], 
    llm_service, 
    max_summary_length: int = None
) -> Dict[str, Any]:
    if not transcript_data:
        return {}
    
    # 按说话人分组收集文本
    speaker_texts = {}
    for item in transcript_data:
        speaker_id = item.get("speaker_id")
        text = item.get("text", "").strip()
        
        if not text or speaker_id is None:
            continue
            
        if isinstance(speaker_id, str) and not speaker_id.isdigit() and len(speaker_id) > 0:
            speaker_key = speaker_id
            speaker_name = speaker_id
        else:
            speaker_key = str(speaker_id)
            speaker_name = item.get("speaker_name") or item.get("employee_id") or None
            
        if speaker_key not in speaker_texts:
            speaker_texts[speaker_key] = {"texts": [], "name": speaker_name or speaker_key, "segments": 0}
            
        speaker_texts[speaker_key]["texts"].append(text)
        speaker_texts[speaker_key]["segments"] += 1

    if not speaker_texts:
        logger.warning("⚠️ 未找到有效的说话人数据")
        return {}

    logger.info(f"📊 开始为 {len(speaker_texts)} 个说话人一次性生成摘要...")

    # 准备所有说话人的内容，用于一次性生成
    speaker_contents = []
    speaker_metadata = {}  # 保存每个说话人的元数据（用于后续格式化）
    
    for speaker_key, data in speaker_texts.items():
        full_text = " ".join(data["texts"])
        word_count = len(full_text)
        speaker_display_name = data.get("name") or speaker_key
        
        # 限制每个说话人的输入长度（避免prompt过长）
        if word_count < 500:
            input_limit, summary_length_hint = word_count, "100-200字"
        elif word_count < 2000:
            input_limit, summary_length_hint = min(word_count, 5000), "200-400字"
        elif word_count < 5000:
            input_limit, summary_length_hint = min(word_count, 8000), "400-800字"
        else:
            input_limit, summary_length_hint = min(word_count, 12000), "800-1500字"
        
        limited_text = full_text[:input_limit]
        
        speaker_contents.append({
            "speaker_key": speaker_key,
            "speaker_name": speaker_display_name,
            "content": limited_text,
            "word_count": word_count,
            "segments": data["segments"],
            "summary_length_hint": summary_length_hint
        })
        
        # 保存元数据
        speaker_metadata[speaker_key] = {
            "name": data.get("name"),
            "word_count": word_count,
            "segments": data["segments"],
            "full_text": full_text
        }

    # 构建一次性生成所有摘要的prompt
    speakers_content_text = ""
    for idx, speaker_info in enumerate(speaker_contents, 1):
        speakers_content_text += f"""
【说话人{idx}】：{speaker_info['speaker_name']}
【该说话人的发言内容】：
{speaker_info['content']}
【建议摘要长度】：{speaker_info['summary_length_hint']}

"""

    if max_summary_length:
        length_instruction = f"每个说话人的摘要控制在{max_summary_length}字以内"
    else:
        length_instruction = "每个说话人的摘要长度根据其内容重要性灵活调整，确保关键信息完整"

    prompt = f"""你是一位拥有10年经验的企业高级高管助理。请为以下**所有说话人的发言内容**一次性生成**发言摘要**（不是会议纪要，只是每个说话人的发言总结），要求：

⚠️⚠️⚠️ **重要说明**：
- 这是**说话人发言摘要**，不是会议纪要
- 需要为**每个说话人**分别生成摘要，只需要总结**该说话人**说了什么，不需要总结整个会议
- 不要生成会议纪要格式，只需要生成每个说话人的发言内容摘要

1. **核心总结**：提取每个说话人的主要观点、业务痛点和后续计划。
2. **决策聚焦**：突出每个说话人提到的关键信息、核心数据和决策点。
3. **篇幅要求**：{length_instruction}。
4. 【红线约束1：静默容错与自动修正】：原始文本由机器识别，包含大量同音错别字、口语化和逻辑跳跃。请你凭借专业知识，在心中自行脑补、修正并翻译成专业的商业/技术术语。绝对禁止在输出结果中提及"文本错误"、"逻辑混乱"或"表述不清"。
5. 【红线约束2：严禁主观评价】：你的唯一职责是"提炼业务价值"。绝对禁止对发言人的身份、语言表达能力、逻辑性进行任何评价、质疑或猜测。禁止输出任何类似"身份存疑"、"矛盾点"等评价性字眼。
6. 【红线约束3：输出格式要求】：
   - **必须严格按照以下JSON格式输出**，不要添加任何其他内容
   - 每个说话人的摘要必须是纯文本（不要用Markdown格式，不要用HTML标签）
   - **绝对禁止在任何字段中输出HTML标签（如<mark>、<p>、<br>、<div>等）**
   - **绝对禁止在任何字段中输出HTML属性（如style=、class=等）**
   - **speaker_name字段的值必须是纯文本字符串，例如："张楠楠"，绝对不要写成：<mark>张楠楠</mark>**
   - **summary字段的值必须是纯文本，只包含文字、标点符号和换行符，不要有任何HTML标记**
   - JSON格式如下（注意：JSON必须是完整的，不要被截断）：
```json
{{
  "summaries": [
    {{
      "speaker_name": "说话人姓名或ID",
      "summary": "该说话人的发言摘要（纯文本，无HTML标签）"
    }},
    ...
  ]
}}
```

【所有说话人的发言内容】：
{speakers_content_text}

请严格按照JSON格式输出所有说话人的摘要："""

    try:
        # 一次性调用LLM生成所有摘要
        # 根据说话人数量动态调整max_tokens：每个说话人约500-1000 tokens
        estimated_tokens = len(speaker_contents) * 800 + 500  # 预留500 tokens给JSON结构
        max_tokens = min(estimated_tokens, 6000)  # 最多6000 tokens
        
        response_text = await asyncio.to_thread(
            llm_service.chat,
            prompt, 
            temperature=0.3, 
            max_tokens=max_tokens
        )
        
        # 解析JSON响应
        response_text = (response_text or "").strip()
        logger.debug(f"📝 LLM返回的原始响应长度: {len(response_text)} 字符")
        
        # 尝试提取JSON（可能被markdown代码块包裹）
        json_text = response_text
        
        # 方法1：尝试提取markdown代码块中的JSON
        if "```json" in json_text:
            parts = json_text.split("```json")
            if len(parts) > 1:
                json_text = parts[1].split("```")[0].strip()
        elif "```" in json_text:
            parts = json_text.split("```")
            if len(parts) > 1:
                json_text = parts[1].split("```")[0].strip()
        
        # 方法2：尝试直接找到JSON对象（从第一个 { 开始）
        if not json_text.startswith("{"):
            first_brace = json_text.find("{")
            if first_brace != -1:
                json_text = json_text[first_brace:].strip()
        
        # 方法3：尝试找到完整的JSON（从 { 到最后一个 }）
        if json_text.startswith("{"):
            # 尝试找到最后一个匹配的 }
            brace_count = 0
            last_brace_idx = -1
            for i, char in enumerate(json_text):
                if char == "{":
                    brace_count += 1
                elif char == "}":
                    brace_count -= 1
                    if brace_count == 0:
                        last_brace_idx = i
                        break
            if last_brace_idx != -1:
                json_text = json_text[:last_brace_idx + 1]
        
        # 解析JSON
        summaries_list = []
        import re
        
        # 预处理：清理JSON字符串值中的HTML标签（避免HTML标签中的引号导致JSON解析失败）
        def clean_html_from_json(text):
            """移除JSON中所有HTML标签，并修复可能被破坏的JSON格式"""
            import re
            # 第一步：移除所有HTML标签（包括在字符串值外的，如speaker_name字段值）
            # 这会处理类似 "speaker_name": <mark>xxx</mark> 的情况
            text = re.sub(r'<[^>]+>', '', text)
            
            # 第二步：修复可能被破坏的JSON格式
            # 例如：修复 "speaker_name": 张楠楠, 变成 "speaker_name": "张楠楠",
            # 匹配类似 "speaker_name": 值, 的模式（值不是字符串的情况）
            # 但要小心，不要匹配已经是字符串的值
            
            # 修复 speaker_name 字段：如果值不是以引号开始，添加引号
            def fix_field(match):
                field_name = match.group(1)
                value = match.group(2).strip()
                # 如果值不是以引号开始，说明可能是被HTML标签包裹的值，需要添加引号
                if not value.startswith('"'):
                    # 移除值末尾可能的逗号或括号
                    value_clean = value.rstrip(',}').strip()
                    return f'"{field_name}": "{value_clean}"'
                return match.group(0)
            
            # 匹配 "speaker_name": 值 或 "summary": 值 的模式
            text = re.sub(r'"speaker_name"\s*:\s*([^",}\]]+?)(\s*[,}\]])', 
                         lambda m: f'"speaker_name": "{m.group(1).strip()}"{m.group(2)}', text)
            text = re.sub(r'"summary"\s*:\s*([^",}\]]+?)(\s*[,}\]])', 
                         lambda m: f'"summary": "{m.group(1).strip()}"{m.group(2)}', text)
            
            return text
        
        # 先尝试清理HTML标签后再解析
        cleaned_json = clean_html_from_json(json_text)
        
        try:
            result_data = json.loads(cleaned_json)
            summaries_list = result_data.get("summaries", [])
            logger.info(f"✅ 成功解析JSON（已清理HTML标签），提取到 {len(summaries_list)} 个说话人摘要")
        except json.JSONDecodeError:
            # 如果清理后还是失败，尝试原始文本
            try:
                result_data = json.loads(json_text)
                summaries_list = result_data.get("summaries", [])
                logger.info(f"✅ 成功解析JSON（原始文本），提取到 {len(summaries_list)} 个说话人摘要")
            except json.JSONDecodeError as e:
                logger.warning(f"⚠️ JSON解析失败: {e}")
                logger.warning(f"⚠️ 尝试解析的JSON文本（前500字符）: {json_text[:500]}")
            
            # 降级方案1：尝试修复常见的JSON格式问题
            try:
                # 修复：处理字符串中未转义的引号（在HTML标签中常见）
                # 但这种方法很危险，可能会破坏正确的JSON，所以先尝试其他方法
                fixed_json = json_text
                
                # 尝试使用更智能的方法：逐字符解析，找到完整的JSON对象
                # 从第一个 { 开始，找到匹配的最后一个 }
                if fixed_json.startswith("{"):
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    last_valid_brace = -1
                    
                    for i, char in enumerate(fixed_json):
                        if escape_next:
                            escape_next = False
                            continue
                        if char == '\\':
                            escape_next = True
                            continue
                        if char == '"' and not escape_next:
                            in_string = not in_string
                        if not in_string:
                            if char == '{':
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    last_valid_brace = i
                                    break
                    
                    if last_valid_brace != -1:
                        fixed_json = fixed_json[:last_valid_brace + 1]
                        try:
                            result_data = json.loads(fixed_json)
                            summaries_list = result_data.get("summaries", [])
                            logger.info(f"✅ 通过智能括号匹配成功提取JSON，获得 {len(summaries_list)} 个摘要")
                        except:
                            pass
            except Exception as e2:
                logger.debug(f"⚠️ JSON修复尝试失败: {e2}")
            
            # 降级方案2：使用正则表达式提取每个说话人的摘要（更可靠）
            if not summaries_list:
                logger.warning("⚠️ 尝试使用正则表达式提取摘要...")
                # 匹配格式：{"speaker_name": "...", "summary": "..."}
                # 注意：summary字段的值可能包含HTML标签和未转义的引号
                pattern = r'\{\s*"speaker_name"\s*:\s*"([^"]+)"\s*,\s*"summary"\s*:\s*"((?:[^"\\]|\\.|"(?=\s*[,\}]))*)'
                
                # 更简单但更可靠的方法：逐行查找每个说话人的摘要
                for speaker_info in speaker_contents:
                    speaker_name = speaker_info['speaker_name']
                    
                    # 方法A：查找 "speaker_name": "xxx" 后面的 "summary": "..." 
                    speaker_pattern = rf'"speaker_name"\s*:\s*"{re.escape(speaker_name)}"'
                    speaker_match = re.search(speaker_pattern, json_text)
                    
                    if speaker_match:
                        # 找到summary字段的开始位置
                        after_speaker = json_text[speaker_match.end():]
                        summary_pattern = r'"summary"\s*:\s*"'
                        summary_match = re.search(summary_pattern, after_speaker)
                        
                        if summary_match:
                            # 从summary的值开始位置提取，直到找到匹配的引号（考虑转义）
                            summary_start = summary_match.end()
                            summary_text = ""
                            i = summary_start
                            while i < len(after_speaker):
                                char = after_speaker[i]
                                if char == '\\' and i + 1 < len(after_speaker):
                                    # 转义字符
                                    summary_text += after_speaker[i:i+2]
                                    i += 2
                                elif char == '"':
                                    # 找到字符串结束
                                    break
                                else:
                                    summary_text += char
                                    i += 1
                            
                            if summary_text:
                                # 清理转义字符
                                summary_text = summary_text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                                summaries_list.append({
                                    "speaker_name": speaker_name,
                                    "summary": summary_text
                                })
                                logger.debug(f"✅ 成功提取说话人 {speaker_name} 的摘要（长度: {len(summary_text)}）")
                    
                    # 方法B：如果方法A失败，尝试简单的字符串查找（作为最后手段）
                    if not any(item.get("speaker_name") == speaker_name for item in summaries_list):
                        # 查找格式：speaker_name": "xxx", ... "summary": "yyy"
                        simple_pattern = rf'"speaker_name"\s*:\s*"{re.escape(speaker_name)}".*?"summary"\s*:\s*"([^"]*(?:\\.[^"]*)*)"'
                        simple_match = re.search(simple_pattern, json_text, re.DOTALL)
                        if simple_match:
                            summary_text = simple_match.group(1)
                            summary_text = summary_text.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                            summaries_list.append({
                                "speaker_name": speaker_name,
                                "summary": summary_text
                            })
                            logger.debug(f"✅ 通过简单模式提取说话人 {speaker_name} 的摘要")
                
                if summaries_list:
                    logger.info(f"✅ 通过正则表达式成功提取 {len(summaries_list)} 个说话人摘要")
        
        # 转换为原来的格式
        summaries = {}
        for summary_item in summaries_list:
            speaker_name = summary_item.get("speaker_name", "")
            summary_text = summary_item.get("summary", "").strip()
            
            # 找到对应的speaker_key
            matched_key = None
            for speaker_info in speaker_contents:
                if speaker_info['speaker_name'] == speaker_name:
                    matched_key = speaker_info['speaker_key']
                    break
            
            if not matched_key:
                # 如果找不到匹配，使用第一个未匹配的
                for key in speaker_texts.keys():
                    if key not in summaries:
                        matched_key = key
                        break
            
            if matched_key and summary_text:
                metadata = speaker_metadata[matched_key]
                
                # 清理摘要文本：移除Markdown代码块、HTML标签、转义字符等
                clean_summary = summary_text.replace("```markdown", "").replace("```", "").strip()
                # 移除所有HTML标签（包括<mark>、<p>、<br>等）
                clean_summary = re.sub(r'<[^>]+>', '', clean_summary)
                # 移除JSON转义字符
                clean_summary = clean_summary.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')
                # 移除多余的空白字符
                clean_summary = re.sub(r'\s+', ' ', clean_summary).strip()
                
                # 转换为HTML5格式
                try:
                    paragraphs = [p.strip() for p in clean_summary.split('\n\n') if p.strip()]
                    if paragraphs:
                        summary_html = ''.join([f'<p>{p.replace(chr(10), "<br>")}</p>' for p in paragraphs])
                    else:
                        summary_html = f'<p>{clean_summary.replace(chr(10), "<br>")}</p>'
                except Exception:
                    summary_html = f'<p>{clean_summary.replace(chr(10), "<br>")}</p>'
                
                # 确定speaker_id和speaker_name
                if matched_key.isdigit() or (isinstance(matched_key, str) and matched_key.startswith("SPEAKER_")):
                    final_speaker_id = speaker_name
                else:
                    final_speaker_id = matched_key
                
                final_speaker_name = metadata.get("name") if metadata.get("name") else None
                
                summaries[matched_key] = {
                    "speaker_id": final_speaker_id,
                    "speaker_name": final_speaker_name,
                    "summary": summary_html,
                    "word_count": metadata["word_count"],
                    "speech_segments": metadata["segments"]
                }
        
        # 如果有些说话人没有生成摘要，使用降级方案
        for speaker_key in speaker_texts.keys():
            if speaker_key not in summaries:
                logger.warning(f"⚠️ 说话人 {speaker_key} 的摘要未生成，使用降级方案")
                metadata = speaker_metadata[speaker_key]
                full_text = metadata["full_text"]
                
                # 生成简单摘要
                if max_summary_length:
                    summary_text = full_text[:max_summary_length] + "..." if len(full_text) > max_summary_length else full_text
                else:
                    summary_text = full_text[:1000] + "..." if len(full_text) > 1000 else full_text
                
                clean_summary = summary_text.replace("```markdown", "").replace("```", "").strip()
                clean_summary = re.sub(r'<[^>]+>', '', clean_summary)
                
                try:
                    paragraphs = [p.strip() for p in clean_summary.split('\n\n') if p.strip()]
                    if paragraphs:
                        summary_html = ''.join([f'<p>{p.replace(chr(10), "<br>")}</p>' for p in paragraphs])
                    else:
                        summary_html = f'<p>{clean_summary.replace(chr(10), "<br>")}</p>'
                except Exception:
                    summary_html = f'<p>{clean_summary.replace(chr(10), "<br>")}</p>'
                
                speaker_display_name = metadata.get("name") or speaker_key
                if speaker_key.isdigit() or (isinstance(speaker_key, str) and speaker_key.startswith("SPEAKER_")):
                    final_speaker_id = speaker_display_name
                else:
                    final_speaker_id = speaker_key
                
                summaries[speaker_key] = {
                    "speaker_id": final_speaker_id,
                    "speaker_name": metadata.get("name"),
                    "summary": summary_html,
                    "word_count": metadata["word_count"],
                    "speech_segments": metadata["segments"]
                }
        
        logger.info(f"✅ 所有说话人摘要一次性生成完成，共 {len(summaries)} 个")
        return summaries
        
    except Exception as e:
        logger.error(f"❌ 一次性生成所有说话人摘要失败: {e}\n{traceback.format_exc()}")
        # 降级方案：返回空字典，让调用方处理
        return {}

# -----------------------------------------------------------------------------
# 2. 并行音频处理模块
# -----------------------------------------------------------------------------
async def handle_audio_parallel_multi_gpu(audio_path: str, is_url: bool, asr_model: str, gpu_devices: List[str]):
    """
    多GPU加速处理单个音频：将音频分段，每段在不同GPU上并行处理
    
    Args:
        audio_path: 音频文件路径或URL
        is_url: 是否为URL
        asr_model: ASR模型名称
        gpu_devices: GPU设备列表（如 ["cuda:0", "cuda:1", ...]）
    """
    from app.services.parallel_processor import map_words_to_speakers, aggregate_by_speaker, parse_rttm
    import subprocess
    import tempfile
    
    try:
        # 1. 获取音频时长和分段
        audio_duration = get_audio_duration(audio_path)
        segment_count = len(gpu_devices)
        segment_duration = audio_duration / segment_count
        
        logger.info(f"📊 音频总时长: {audio_duration:.1f}秒，分为 {segment_count} 段，每段约 {segment_duration:.1f}秒")
        
        # 2. 如果音频时长太短，降级为单GPU处理
        if audio_duration < 60:  # 小于1分钟，不值得分段
            logger.info("⚠️ 音频时长过短，降级为单GPU处理")
            return await handle_audio_parallel(audio_path, is_url, asr_model, gpu_devices[0] if gpu_devices else None)
        
        # 3. 下载音频到本地（如果是URL）
        local_audio_path = audio_path
        temp_audio_file = None
        if is_url:
            temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            async with httpx.AsyncClient() as client:
                async with client.stream("GET", audio_path, timeout=300) as resp:
                    resp.raise_for_status()
                    with open(temp_audio_file.name, "wb") as f:
                        async for chunk in resp.aiter_bytes(chunk_size=8192):
                            f.write(chunk)
            local_audio_path = temp_audio_file.name
        
        # 4. 分段处理：每段在不同GPU上并行处理
        segment_tasks = []
        for i, gpu_device in enumerate(gpu_devices):
            start_time = i * segment_duration
            end_time = (i + 1) * segment_duration if i < len(gpu_devices) - 1 else audio_duration
            
            # 提取音频片段
            segment_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            try:
                # 使用ffmpeg提取片段
                cmd = [
                    "ffmpeg", "-i", local_audio_path,
                    "-ss", str(start_time),
                    "-t", str(end_time - start_time),
                    "-acodec", "copy",
                    "-y", segment_file.name
                ]
                subprocess.run(cmd, check=True, capture_output=True, timeout=60)
                
                # 创建处理任务
                segment_tasks.append(
                    handle_audio_parallel(segment_file.name, False, asr_model, gpu_device)
                )
            except Exception as e:
                logger.error(f"❌ 提取音频片段 {i} 失败: {e}")
                segment_file.close()
                os.unlink(segment_file.name)
        
        # 5. 并行处理所有片段
        segment_results = await asyncio.gather(*segment_tasks, return_exceptions=True)
        
        # 6. 合并结果
        all_words = []
        all_transcript_data = []
        time_offset = 0.0
        
        for i, result in enumerate(segment_results):
            if isinstance(result, Exception):
                logger.error(f"❌ 片段 {i} 处理失败: {result}")
                continue
            
            if not result or len(result) < 2:
                continue
            
            seg_words, seg_transcript, seg_task_id = result[0], result[1], result[2] if len(result) > 2 else None
            
            if not seg_transcript:
                continue
            
            # 调整时间戳（加上偏移）
            if seg_words:
                for word in seg_words:
                    word["start"] = word.get("start", 0) + time_offset
                    word["end"] = word.get("end", 0) + time_offset
            
            for item in seg_transcript:
                item["start_time"] = int((item.get("start_time", 0) / 1000.0 + time_offset) * 1000)
                item["end_time"] = int((item.get("end_time", 0) / 1000.0 + time_offset) * 1000)
            
            if seg_words:
                all_words.extend(seg_words)
            all_transcript_data.extend(seg_transcript)
            
            # 更新偏移
            if seg_transcript:
                last_end = max(item.get("end_time", 0) for item in seg_transcript) / 1000.0
                time_offset = last_end
        
        # 7. 合并结果并重新聚合（如果需要）
        # 由于分段处理，需要重新进行说话人分离
        # 这里简化处理：直接使用合并后的transcript
        if not all_transcript_data:
            logger.warning("⚠️ 多GPU处理结果为空")
            return None, None, None
        
        raw_text = "".join([item.get("text", "") for item in all_transcript_data])
        
        # 清理临时文件
        if temp_audio_file:
            try:
                os.unlink(temp_audio_file.name)
            except:
                pass
        
        for task in segment_tasks:
            # 清理片段文件（在handle_audio_parallel中应该已经处理）
            pass
        
        logger.info(f"✅ 多GPU加速处理完成，共处理 {len(all_transcript_data)} 个片段")
        return raw_text, all_transcript_data, None
        
    except Exception as e:
        logger.error(f"❌ 多GPU加速处理失败: {e}")
        # 降级为单GPU处理
        return await handle_audio_parallel(audio_path, is_url, asr_model, gpu_devices[0] if gpu_devices else None)


async def handle_audio_parallel(audio_path: str, is_url: bool, asr_model: str, gpu_device: Optional[str] = None):
    """
    封装并行处理逻辑 (全异步流式 I/O)
    
    Args:
        audio_path: 音频文件路径或URL
        is_url: 是否为URL
        asr_model: ASR模型名称
        gpu_device: GPU设备字符串（如 "cuda:0"），如果为None则从GPU池获取
    """
    from app.services.parallel_processor import map_words_to_speakers, aggregate_by_speaker, parse_rttm
    from app.services.gpu_pool import get_gpu_pool
    
    # 验证音频文件（如果不是URL）
    if not is_url:
        if not os.path.exists(audio_path):
            logger.error(f"❌ 音频文件不存在: {audio_path}")
            return None, None, None
        
        file_size = os.path.getsize(audio_path)
        if file_size == 0:
            logger.error(f"❌ 音频文件大小为0: {audio_path}")
            return None, None, None
        
        if file_size < 1024:  # 小于1KB，可能是无效文件
            logger.warning(f"⚠️ 音频文件过小 ({file_size} 字节): {audio_path}，可能无效")
        
        logger.debug(f"📁 音频文件验证通过: {audio_path}, 大小: {file_size} 字节")
    
    # 基础服务URL
    base_funasr_url = os.getenv("FUNASR_SERVICE_URL", "")
    base_pyannote_url = os.getenv("PYANNOTE_SERVICE_URL", "")
    
    # 如果没有指定GPU，从GPU池获取
    acquired_gpu = None
    if gpu_device is None:
        try:
            gpu_pool = get_gpu_pool()
            acquired_gpu = await gpu_pool.acquire_gpu(strategy="round_robin")
            if acquired_gpu:
                logger.info(f"🎯 从GPU池获取GPU: {acquired_gpu}")
                gpu_device = acquired_gpu
        except Exception as e:
            logger.warning(f"⚠️ 获取GPU失败，使用默认设备: {e}")
            gpu_device = None
    
    # 根据GPU设备选择对应的服务URL（支持多实例部署）
    funasr_url = base_funasr_url
    pyannote_url = base_pyannote_url
    
    # 如果指定了GPU设备，尝试选择对应的服务实例
    gpu_id = None
    if gpu_device and gpu_device.startswith("cuda:"):
        try:
            gpu_id = int(gpu_device.split(":")[1])
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ GPU设备解析失败: {e}")
    
    # 如果GPU池为空（主服务容器没有GPU），但配置了多实例服务，且没有传入gpu_device，使用轮询策略
    # 注意：如果外部已经传入了 gpu_device（如从任务索引计算得出），优先使用传入的
    if gpu_id is None and base_funasr_url and not gpu_device:
        try:
            from urllib.parse import urlparse
            parsed = urlparse(base_funasr_url)
            # 如果 base_funasr_url 是 funasr-gpu0 格式，说明是多实例部署
            if "funasr-gpu" in parsed.netloc:
                # 尝试从 docker-compose 环境变量获取 GPU 数量，默认使用 7 个（funasr-gpu0 到 funasr-gpu6）
                max_gpu_instances = int(os.getenv("MAX_FUNASR_GPU_INSTANCES", "7"))
                # 使用简单的轮询：根据当前时间戳选择 GPU（作为兜底策略）
                import time
                seed = int(time.time() * 1000000) % max_gpu_instances
                gpu_id = seed
                logger.info(f"🔄 GPU池为空且未传入GPU设备，使用时间戳轮询选择 GPU 实例: funasr-gpu{gpu_id}")
        except Exception as e:
            logger.warning(f"⚠️ 轮询策略选择GPU失败，使用默认服务URL: {e}")
    
    # 根据 GPU ID 构建服务 URL
    if gpu_id is not None:
        try:
            from urllib.parse import urlparse
            if base_funasr_url:
                parsed = urlparse(base_funasr_url)
                if "localhost" in parsed.netloc or "127.0.0.1" in parsed.netloc:
                    funasr_url = f"{parsed.scheme}://{parsed.netloc.split(':')[0]}:{8002 + gpu_id}"
                else:
                    # 使用服务名：funasr-gpu0, funasr-gpu1, ...
                    funasr_url = f"{parsed.scheme}://funasr-gpu{gpu_id}:{parsed.port or 8002}"
                    # 如果原URL有路径（如 /api/v1），保留路径
                    if parsed.path:
                        funasr_url += parsed.path
            if base_pyannote_url:
                parsed = urlparse(base_pyannote_url)
                if "localhost" in parsed.netloc or "127.0.0.1" in parsed.netloc:
                    pyannote_url = f"{parsed.scheme}://{parsed.netloc.split(':')[0]}:{8100 + gpu_id}"
                else:
                    pyannote_url = f"{parsed.scheme}://pyannote-gpu{gpu_id}:{parsed.port or 8100}"
                    if parsed.path:
                        pyannote_url += parsed.path
            logger.info(f"🎯 使用服务实例 (GPU {gpu_id}) - FunASR: {funasr_url}, Pyannote: {pyannote_url}")
        except (ValueError, IndexError) as e:
            logger.warning(f"⚠️ GPU服务URL构建失败，使用默认服务URL: {e}")
    
    # 确保在函数结束时释放GPU
    try:
        async def run_funasr():
            url = f"{funasr_url}/transcribe/word-level"
            params = {"hotword": ""}
            # 如果指定了GPU设备，传递给FunASR服务
            if gpu_device:
                params["device"] = gpu_device
            async with httpx.AsyncClient(timeout=600.0) as client:
                try:
                    if is_url:
                        resp = await client.post(url, data={"audio_url": audio_path, **params})
                    else:
                        with open(audio_path, "rb") as f:
                            resp = await client.post(url, files={"file": f}, data=params)
                    
                    # 检查 HTTP 状态码
                    if resp.status_code != 200:
                        error_text = resp.text[:500]  # 只取前500字符
                        logger.error(f"❌ FunASR 服务返回错误状态码 {resp.status_code}: {error_text}")
                        raise ValueError(f"FunASR 服务返回错误: HTTP {resp.status_code}")
                    
                    # 尝试解析 JSON
                    try:
                        resp_json = resp.json()
                    except Exception as json_err:
                        error_text = resp.text[:500]
                        logger.error(f"❌ FunASR 响应不是有效的 JSON: {json_err}")
                        logger.error(f"   响应内容（前500字符）: {error_text}")
                        logger.error(f"   响应状态码: {resp.status_code}")
                        logger.error(f"   请求URL: {url}")
                        raise ValueError(f"FunASR 响应解析失败: {json_err}")
                    
                    words = resp_json.get("words", [])
                    # 提取流水号
                    task_id = resp_json.get("task_id") or resp_json.get("request_id")
                    if not task_id and "data" in resp_json:
                        task_id = resp_json["data"].get("task_id") or resp_json["data"].get("request_id")
                    # 如果没有流水号，生成一个
                    if not task_id:
                        task_id = str(uuid.uuid4())
                        logger.info(f"📝 并行处理FunASR生成唯一标识: {task_id}")
                    else:
                        task_id = str(task_id)
                        logger.info(f"📝 并行处理FunASR返回流水号: {task_id}")
                    
                    # 如果 words 为空，记录完整响应以便诊断
                    if not words:
                        logger.warning(f"⚠️ FunASR 返回空 words，完整响应: {resp_json}")
                    
                    return words, task_id
                except httpx.TimeoutException:
                    logger.error(f"❌ FunASR 请求超时: {url}")
                    raise
                except httpx.RequestError as e:
                    logger.error(f"❌ FunASR 请求失败: {e}")
                    raise

        async def run_pyannote():
            url = f"{pyannote_url}/rttm"
            params = {}
            # 如果指定了GPU设备，传递给Pyannote服务
            if gpu_device:
                params["device"] = gpu_device
            async with httpx.AsyncClient(timeout=600.0) as client:
                try:
                    if is_url:
                        resp = await client.post(url, data={"audio_url": audio_path, **params})
                    else:
                        with open(audio_path, "rb") as f:
                            resp = await client.post(url, files={"file": f}, data=params)
                    
                    # 检查 HTTP 状态码
                    if resp.status_code != 200:
                        error_text = resp.text[:1000]  # 增加错误文本长度，便于诊断
                        logger.error(f"❌ Pyannote 服务返回错误状态码 {resp.status_code}")
                        logger.error(f"   请求URL: {url}")
                        logger.error(f"   错误响应内容: {error_text}")
                        logger.error(f"   响应头: {dict(resp.headers)}")
                        raise ValueError(f"Pyannote 服务返回错误: HTTP {resp.status_code}")
                    
                    # 尝试解析 JSON
                    try:
                        resp_json = resp.json()
                    except Exception as json_err:
                        error_text = resp.text[:500]
                        logger.error(f"❌ Pyannote 响应不是有效的 JSON: {json_err}")
                        logger.error(f"   响应内容（前500字符）: {error_text}")
                        logger.error(f"   响应状态码: {resp.status_code}")
                        logger.error(f"   请求URL: {url}")
                        raise ValueError(f"Pyannote 响应解析失败: {json_err}")
                    
                    rttm = resp_json.get("rttm", "")
                    
                    # 如果 rttm 为空，记录完整响应以便诊断
                    if not rttm:
                        logger.warning(f"⚠️ Pyannote 返回空 rttm，完整响应: {resp_json}")
                    
                    return rttm
                except httpx.TimeoutException:
                    logger.error(f"❌ Pyannote 请求超时: {url}")
                    raise
                except httpx.RequestError as e:
                    logger.error(f"❌ Pyannote 请求失败: {e}")
                    raise

        try:
            funasr_result, f_rttm = await asyncio.gather(run_funasr(), run_pyannote(), return_exceptions=True)
            
            # 检查是否有异常
            if isinstance(funasr_result, Exception):
                logger.error(f"❌ FunASR 调用异常: {funasr_result}")
                return None, None, None
            if isinstance(f_rttm, Exception):
                logger.error(f"❌ Pyannote 调用异常: {f_rttm}")
                return None, None, None
            
            # funasr_result 现在是 (words, task_id) 元组
            f_words, funasr_task_id = funasr_result if isinstance(funasr_result, tuple) else (funasr_result, None)

            if not f_words or not f_rttm:
                logger.warning("⚠️ ASR 或 Pyannote 服务返回为空")
                logger.warning(f"   FunASR words 长度: {len(f_words) if f_words else 0}, 类型: {type(f_words)}")
                logger.warning(f"   Pyannote rttm 长度: {len(f_rttm) if f_rttm else 0}, 类型: {type(f_rttm)}")
                logger.warning(f"   音频路径: {audio_path}")
                logger.warning(f"   FunASR URL: {funasr_url}")
                logger.warning(f"   Pyannote URL: {pyannote_url}")
                logger.warning(f"   GPU设备: {gpu_device}")
                return None, None, None
        except Exception as gather_err:
            logger.error(f"❌ 并行调用服务时发生异常: {gather_err}")
            return None, None, None

        rttm_segments = parse_rttm(f_rttm)
        mapped = map_words_to_speakers(f_words, rttm_segments)
        transcript_data = aggregate_by_speaker(mapped)

        # 记录并行聚合后的原始 speaker_id 样例
        if transcript_data:
            sample_speaker_ids = [item.get("speaker_id", "UNKNOWN") for item in transcript_data[:10]]
            logger.info(f"🔎 并行聚合后原始 speaker_id 样例: {sample_speaker_ids}")

        def speaker_str_to_int(speaker_str: str) -> Optional[int]:
            if not speaker_str: return None
            try:
                if isinstance(speaker_str, int): return speaker_str
                if isinstance(speaker_str, str) and speaker_str.startswith("SPEAKER_"):
                    num_str = speaker_str.replace("SPEAKER_", "").strip()
                    return int(num_str) if num_str.isdigit() else None
                return int(speaker_str) if str(speaker_str).isdigit() else None
            except (ValueError, AttributeError):
                return None
        
        def is_valid_text(text: str) -> bool:
            if not text:
                return False
            clean_text = re.sub(r'[^\w\u4e00-\u9fa5a-zA-Z0-9]', '', text)
            return len(clean_text) > 0

        formatted_data = []
        for item in transcript_data:
            text = item.get("text", "").strip()
            if not is_valid_text(text):
                logger.debug(f"🗑️ 过滤掉无效的噪音片段: [{item.get('start')} - {item.get('end')}]")
                continue
                
            # Pyannote 返回的时间是秒，需要转换为毫秒
            start_sec = float(item.get("start", 0.0))
            end_sec = float(item.get("end", start_sec))
            formatted_data.append({
                "text": text,
                "start_time": int(start_sec * 1000),
                "end_time": int(end_sec * 1000),
                "speaker_id": speaker_str_to_int(item.get("speaker_id", "SPEAKER_00"))
            })
        
        return "".join([i["text"] for i in formatted_data]), formatted_data, funasr_task_id
    except Exception as e:
        logger.error(f"❌ 并行请求服务失败: {e}")
        return None, None, None
    finally:
        # 释放GPU资源
        if acquired_gpu:
            try:
                from app.services.gpu_pool import get_gpu_pool
                gpu_pool = get_gpu_pool()
                await gpu_pool.release_gpu(acquired_gpu)
                logger.debug(f"🔓 释放GPU: {acquired_gpu}")
            except Exception as e:
                logger.warning(f"⚠️ 释放GPU失败: {e}")

# -----------------------------------------------------------------------------
# 3. 主接口 API
# -----------------------------------------------------------------------------
MAX_CONCURRENT_PROCESS = None

# 全局任务计数器（用于跨请求的负载均衡）
_global_task_counter = 0
_task_counter_lock = asyncio.Lock()

async def get_next_task_index() -> int:
    """获取下一个全局任务索引（线程安全）"""
    global _global_task_counter
    async with _task_counter_lock:
        current = _global_task_counter
        _global_task_counter += 1
        return current

def _init_concurrent_semaphore():
    """初始化并发控制Semaphore：根据GPU池容量动态设置"""
    global MAX_CONCURRENT_PROCESS
    if MAX_CONCURRENT_PROCESS is None:
        try:
            from app.services.gpu_pool import get_gpu_pool
            gpu_pool = get_gpu_pool(max_concurrent_per_gpu=1)
            total_capacity = gpu_pool.get_total_concurrent_capacity()
            if total_capacity > 0:
                MAX_CONCURRENT_PROCESS = asyncio.Semaphore(total_capacity)
                logger.info(f"🚀 根据GPU池容量初始化并发控制: {total_capacity} (检测到 {gpu_pool.get_gpu_count()} 张GPU)")
            else:
                # 没有GPU，使用CPU模式，设置较小的并发数
                MAX_CONCURRENT_PROCESS = asyncio.Semaphore(1)
                logger.info("⚠️ 未检测到GPU，使用CPU模式，并发数: 1")
        except Exception as e:
            logger.warning(f"⚠️ GPU池初始化失败，使用默认并发数: {e}")
            MAX_CONCURRENT_PROCESS = asyncio.Semaphore(1)

@router.post("/process")
async def process_meeting_audio(
    files: Optional[List[UploadFile]] = File(None),
    file_paths: Optional[str] = Form(None),
    audio_urls: Optional[str] = Form(None),
    audio_id: Optional[int] = Form(None),
    document_urls: Optional[str] = Form(None),
    text_content: Optional[str] = Form(None),
    template: str = Form("default"),
    user_requirement: Optional[str] = Form(None),
    history_meeting_ids: Optional[str] = Form(None),
    history_mode: str = Form("auto"),
    llm_model: str = Form("auto"),
    llm_temperature: float = Form(0.3), # 建议降为0.3保证格式严格
    llm_max_tokens: int = Form(2000),
    asr_model: str = Form("auto"),
    use_transcript_for_llm: bool = Form(True),
    rebuild: Optional[str] = Form(None),  # 重建模式：前端将之前的 transcript JSON 传回，用于重新生成纪要和摘要
    existing_minutes_html: Optional[str] = Form(
        None,
        description="（可选）现有 H5/HTML 会议纪要内容，用于在原纪要基础上进行修改/润色"
    ),
):
    # 初始化并发控制（如果还未初始化）
    _init_concurrent_semaphore()

    temp_to_clean = []
    raw_text, transcript_data = "", []
    # 标记当前是否为“仅文本模式”（未提供音频/转写，只给了文字说明）
    text_only_mode = False

    try:
        # 获取处理令牌（根据GPU池容量动态控制）
        async with MAX_CONCURRENT_PROCESS:
            logger.info("🚦 获取到处理令牌，开始执行任务...")
            
            reference_document_text = None
            
            # --- 处理文档附件URL ---
            if document_urls:
                url_list = [url.strip().strip('"') for url in document_urls.split(',') if url.strip()]
                logger.info(f"📎 收到 {len(url_list)} 个文档URL")
                all_document_texts = []
                for idx, doc_url in enumerate(url_list):
                    if not doc_url.startswith(("http://", "https://")):
                        continue
                    try:
                        logger.info(f"📥 [{idx+1}/{len(url_list)}] 开始下载文档: {doc_url}")
                        async with httpx.AsyncClient() as client:
                            async with client.stream("GET", doc_url, timeout=300) as resp:
                                resp.raise_for_status()
                                filename = os.path.basename(doc_url)
                                content_disposition = resp.headers.get('Content-Disposition', '')
                                if 'filename=' in content_disposition:
                                    match = re.search(r'filename=["\']?([^"\']+)["\']?', content_disposition)
                                    if match:
                                        filename = match.group(1)
                                
                                if not os.path.splitext(filename)[1]:
                                    content_type = resp.headers.get('Content-Type', '')
                                    if 'word' in content_type.lower() or 'document' in content_type.lower():
                                        filename += '.docx'
                                    elif 'pdf' in content_type.lower():
                                        filename += '.pdf'
                                    elif 'text' in content_type.lower():
                                        filename += '.txt'
                                    else:
                                        filename += '.docx'
                                
                                doc_path = settings.TEMP_DIR / f"doc_{uuid.uuid4().hex}_{filename}"
                                with open(doc_path, "wb") as f:
                                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                                        f.write(chunk)
                                        
                        temp_to_clean.append(str(doc_path))
                        logger.info(f"✅ [{idx+1}/{len(url_list)}] 文档下载完成: {filename}")
                        
                        # 假设 document_service 已被正确导入
                        doc_text = document_service.extract_text_from_file(str(doc_path))
                        if doc_text:
                            all_document_texts.append(f"【文档 {idx+1}: {filename}】\n{doc_text}\n")
                        else:
                            logger.warning(f"⚠️ [{idx+1}/{len(url_list)}] 文档解析失败或为空: {filename}")
                    except Exception as e:
                        logger.error(f"❌ [{idx+1}/{len(url_list)}] 文档下载/解析失败: {doc_url}, 错误: {e}")
                        continue
                
                if all_document_texts:
                    reference_document_text = "\n\n".join(all_document_texts)
                else:
                    logger.warning("⚠️ 所有文档处理失败，将不使用参考文档")

            # --- 输入源解析与预处理 ---
            # 优先级：rebuild > text_content > 音频/URL
            if rebuild:
                # rebuild 模式：前端/后端将之前的 transcript JSON 传回，用于重新生成纪要和摘要
                try:
                    logger.info(f"🔍 开始解析 rebuild 参数，长度: {len(rebuild)} 字符")
                    parsed = json.loads(rebuild)
                    logger.info(f"✅ JSON 解析成功，类型: {type(parsed).__name__}")
                    
                    # 兼容两种格式：直接是数组，或是包含 transcript 字段的对象
                    if isinstance(parsed, dict):
                        logger.info(f"📦 检测到字典格式，键: {list(parsed.keys())}")
                        if "transcript" in parsed:
                            items = parsed.get("transcript") or []
                            logger.info(f"✅ 从 transcript 字段提取数据，类型: {type(items).__name__}, 长度: {len(items) if isinstance(items, list) else 'N/A'}")
                        else:
                            # 尝试其他可能的字段名
                            possible_keys = ["transcripts", "data", "items", "segments"]
                            items = None
                            for key in possible_keys:
                                if key in parsed:
                                    items = parsed.get(key) or []
                                    logger.info(f"✅ 从 {key} 字段提取数据")
                                    break
                            if items is None:
                                # 如果字典只有一个键，且值是数组，则使用该值
                                if len(parsed) == 1:
                                    key = list(parsed.keys())[0]
                                    value = parsed[key]
                                    if isinstance(value, list):
                                        items = value
                                        logger.info(f"✅ 使用唯一键 {key} 的值作为 transcript 数据")
                                    else:
                                        raise ValueError(f"字典中未找到 transcript 字段，且唯一键 {key} 的值不是数组")
                                else:
                                    raise ValueError(f"字典中未找到 transcript 字段，可用字段: {list(parsed.keys())}")
                    else:
                        items = parsed
                        logger.info(f"📦 直接使用解析结果作为数组，类型: {type(items).__name__}")
                    
                    if not isinstance(items, list):
                        raise ValueError(f"rebuild 内容不是有效的数组，实际类型: {type(items).__name__}")

                    if len(items) == 0:
                        raise ValueError("rebuild 内容中的 transcript 数组为空")

                    logger.info(f"📊 开始处理 {len(items)} 个 transcript 项")
                    rebuilt_transcript: List[Dict[str, Any]] = []
                    skipped_count = 0
                    for idx, item in enumerate(items):
                        if not isinstance(item, dict):
                            logger.warning(f"⚠️ 跳过第 {idx+1} 项：不是字典类型，实际类型: {type(item).__name__}")
                            skipped_count += 1
                            continue
                        text = (item.get("text") or "").strip()
                        if not text:
                            logger.warning(f"⚠️ 跳过第 {idx+1} 项：text 字段为空或缺失，可用字段: {list(item.keys())}")
                            skipped_count += 1
                            continue
                        # rebuild 参数中的时间可能是秒或毫秒，统一转换为毫秒
                        start_raw = item.get("start_time", 0.0)
                        end_raw = item.get("end_time", start_raw)
                        # 如果时间小于1000，可能是秒，需要转换；否则认为是毫秒
                        start_ms = int(float(start_raw) * 1000) if float(start_raw) < 1000 else int(start_raw)
                        end_ms = int(float(end_raw) * 1000) if float(end_raw) < 1000 else int(end_raw)
                        speaker_id = item.get("speaker_id")
                        audio_id_item = item.get("audio_id")
                        asr_task_id_item = item.get("asr_task_id")
                        rebuilt_transcript.append(
                            {
                                "text": text,
                                "start_time": start_ms,
                                "end_time": end_ms,
                                "speaker_id": speaker_id,
                                "audio_id": audio_id_item,
                                "asr_task_id": asr_task_id_item,
                            }
                        )

                    if not rebuilt_transcript:
                        error_msg = f"rebuild 内容中未找到有效的 transcript 数据。共处理 {len(items)} 项，跳过 {skipped_count} 项（非字典或 text 为空）"
                        if len(items) > 0 and isinstance(items[0], dict):
                            error_msg += f"。第一项示例字段: {list(items[0].keys())}"
                        raise ValueError(error_msg)

                    # 保持时间轴连续
                    make_transcript_timestamps_continuous(rebuilt_transcript)

                    transcript_data = rebuilt_transcript
                    raw_text = "".join(seg["text"] for seg in rebuilt_transcript)
                    logger.info(f"🧱 重建模式：已从 rebuild 中恢复 transcript，共 {len(transcript_data)} 段，总文本长度: {len(raw_text)} 字符")
                except json.JSONDecodeError as e:
                    logger.error(f"❌ rebuild JSON 解析失败: {e}")
                    logger.error(f"📄 rebuild 内容前200字符: {rebuild[:200]}")
                    raise HTTPException(status_code=400, detail=f"rebuild 内容不是有效的 JSON: {e}")
                except Exception as e:
                    logger.error(f"❌ rebuild 参数解析失败: {e}")
                    logger.error(f"📄 rebuild 内容前200字符: {rebuild[:200]}")
                    raise HTTPException(status_code=400, detail=f"无效的 rebuild 内容: {e}")

            elif text_content:
                # 仅文本模式：没有上传音频 / 没有提供转写，只给了一段文字说明
                raw_text = text_content
                transcript_data = []
                text_only_mode = True
            else:
                audio_paths, is_url_list, audio_names, original_urls = [], [], [], []
                
                if files:
                    logger.info(f"📁 收到 {len(files)} 个音频文件上传")
                    for idx, f in enumerate(files):
                        p = settings.TEMP_DIR / f"multi_{uuid.uuid4().hex}_{idx}_{f.filename}"
                        with open(p, "wb") as b:
                            shutil.copyfileobj(f.file, b)
                        temp_to_clean.append(str(p))
                        audio_paths.append(str(p))
                        is_url_list.append(False)
                        original_urls.append(None)  # 文件上传没有原始URL
                        audio_names.append(f.filename or f"file_{idx+1}")
                        
                elif audio_urls:
                    # 支持两种格式：
                    # 1. JSON格式：'[{"audio_id": "11", "audio_url": "http://..."}]' 或 '[{"audio_id": 11, "audio_url": "http://..."}]'
                    # 2. 简单格式：'http://url1,http://url2'（逗号分隔的URL列表）
                    audio_items = []
                    try:
                        # 尝试解析为JSON格式
                        parsed = json.loads(audio_urls)
                        if isinstance(parsed, list):
                            # JSON数组格式
                            for item in parsed:
                                if isinstance(item, dict):
                                    audio_url = item.get("audio_url") or item.get("audioUrl") or item.get("url")
                                    audio_item_id = item.get("audio_id") or item.get("audioId")
                                    if audio_url:
                                        audio_items.append({"url": audio_url, "id": audio_item_id})
                                elif isinstance(item, str):
                                    # 兼容：如果数组元素是字符串，当作URL处理
                                    audio_items.append({"url": item, "id": None})
                        else:
                            # 如果不是数组，当作单个URL处理
                            if isinstance(parsed, str):
                                audio_items.append({"url": parsed, "id": None})
                    except (json.JSONDecodeError, ValueError):
                        # 不是JSON格式，当作逗号分隔的URL列表处理
                        url_list = [url.strip().strip('"') for url in audio_urls.split(',') if url.strip()]
                        for url in url_list:
                            audio_items.append({"url": url, "id": None})
                    
                    # 处理每个音频项
                    for idx, audio_item in enumerate(audio_items):
                        raw_url = audio_item["url"]
                        audio_item_id = audio_item["id"]
                        
                        if raw_url.startswith("http"):
                            # 保存原始URL（用于腾讯云ASR等需要URL的服务）
                            original_urls.append(raw_url)
                            
                            # 检查ASR服务类型，如果是腾讯云，不需要下载，直接使用URL
                            if asr_model == "tencent":
                                logger.info(f"🔗 使用腾讯云ASR，直接使用URL，不下载: {raw_url}")
                                audio_paths.append(raw_url)
                                is_url_list.append(True)  # 标记为URL
                                base_name = os.path.basename(raw_url.split('?', 1)[0]) or f"url_{idx+1}"
                                audio_names.append(audio_item_id if audio_item_id is not None else base_name)
                            else:
                                # 其他ASR服务（如FunASR）可能需要本地文件，先下载
                                ext = os.path.splitext(raw_url)[1] or ".m4a"
                                local_temp_path = settings.TEMP_DIR / f"url_dl_{uuid.uuid4().hex}{ext}"
                                try:
                                    async with httpx.AsyncClient() as client:
                                        async with client.stream("GET", raw_url, timeout=300) as resp:
                                            resp.raise_for_status()
                                            with open(local_temp_path, "wb") as f:
                                                async for chunk in resp.aiter_bytes(chunk_size=8192):
                                                    f.write(chunk)
                                    temp_to_clean.append(str(local_temp_path))
                                    audio_paths.append(str(local_temp_path))
                                    is_url_list.append(False)
                                    base_name = os.path.basename(raw_url.split('?', 1)[0]) or f"url_{idx+1}"
                                    audio_names.append(audio_item_id if audio_item_id is not None else base_name)
                                except Exception as e:
                                    raise HTTPException(status_code=400, detail=f"无法下载音频URL: {raw_url}, 错误: {str(e)}")
                        else:
                            audio_paths.append(raw_url)
                            is_url_list.append(False)
                            original_urls.append(None)
                            audio_names.append(audio_item_id if audio_item_id is not None else os.path.basename(raw_url) or raw_url)
                            
                elif file_paths:
                    path_list = [path.strip() for path in file_paths.split(',') if path.strip()]
                    audio_paths.extend(path_list)
                    is_url_list.extend([False] * len(path_list))
                    original_urls.extend([None] * len(path_list))
                    audio_names.extend([os.path.basename(p) or p for p in path_list])
                    
                if not audio_paths:
                    raise HTTPException(status_code=400, detail="未提供有效的音频")
                
                logger.info(f"🎵 开始处理 {len(audio_paths)} 个音频...")
                all_raw_texts, all_transcript_data, time_offset = [], [], 0.0
                funasr_url = os.getenv("FUNASR_SERVICE_URL")
                pyannote_url = os.getenv("PYANNOTE_SERVICE_URL")
                
                # --- 智能调度策略：根据任务数量选择GPU分配方式 ---
                from app.services.gpu_pool import get_gpu_pool
                gpu_pool = get_gpu_pool()
                total_gpus = gpu_pool.get_gpu_count()
                task_count = len(audio_paths)
                
                # 调度策略：
                # 1. 单个任务：使用多GPU加速（如果音频较长）
                # 2. 多个任务 <= GPU数：每个任务分配一张GPU
                # 3. 多个任务 > GPU数：一卡一任务，轮询分配
                use_multi_gpu_acceleration = False
                if task_count == 1 and total_gpus > 1:
                    # 单个任务：检查音频时长，决定是否使用多GPU加速
                    try:
                        audio_duration = get_audio_duration(audio_paths[0])
                        # 如果音频时长超过10分钟，使用多GPU加速
                        if audio_duration > 600:  # 10分钟 = 600秒
                            use_multi_gpu_acceleration = True
                            logger.info(f"🚀 检测到单个长音频任务（{audio_duration:.1f}秒），启用多GPU加速模式（使用 {min(total_gpus, 4)} 张GPU）")
                    except Exception as e:
                        logger.warning(f"⚠️ 无法获取音频时长，使用单GPU模式: {e}")
                
                if use_multi_gpu_acceleration:
                    # 多GPU加速模式：将音频分段，每段在不同GPU上处理
                    logger.info(f"⚡ 多GPU加速模式：将使用 {min(total_gpus, 4)} 张GPU并行处理单个音频")
                    # 这里会在 handle_audio_parallel_multi_gpu 中实现
                else:
                    logger.info(f"📊 标准模式：{task_count} 个任务，{total_gpus} 张GPU，采用一卡一任务策略")
            
                # --- 循环处理所有音频 ---
                for idx, audio_path in enumerate(audio_paths):
                    is_url = is_url_list[idx] if idx < len(is_url_list) else False
                    original_url = original_urls[idx] if idx < len(original_urls) else None
                    audio_name = audio_names[idx] if idx < len(audio_names) else os.path.basename(audio_path)
                    logger.info(f"🎤 [{idx+1}/{len(audio_paths)}] 处理音频: {audio_name} ({audio_path})")
                    
                    single_raw_text = ""
                    single_transcript_data = []
                    asr_task_id = None
                    
                    if funasr_url and pyannote_url and asr_model == "funasr":
                        logger.info(f"🚀 启动并行处理引擎...")
                        # 根据调度策略选择处理方式
                        if use_multi_gpu_acceleration and idx == 0:  # 只对第一个（也是唯一的）任务使用多GPU
                            # 多GPU加速模式
                            try:
                                # 获取多张GPU（最多4张，避免过度分段）
                                gpu_devices = await gpu_pool.acquire_multiple_gpus(count=min(total_gpus, 4))
                                if gpu_devices:
                                    logger.info(f"⚡ 多GPU加速：使用 {len(gpu_devices)} 张GPU并行处理")
                                    result = await handle_audio_parallel_multi_gpu(audio_path, is_url, asr_model, gpu_devices)
                                    await gpu_pool.release_multiple_gpus(gpu_devices)
                                else:
                                    # 降级为单GPU
                                    async with gpu_pool.get_gpu_context() as gpu_device:
                                        result = await handle_audio_parallel(audio_path, is_url, asr_model, gpu_device)
                            except Exception as e:
                                logger.warning(f"⚠️ 多GPU加速失败，降级为单GPU: {e}")
                                async with gpu_pool.get_gpu_context() as gpu_device:
                                    result = await handle_audio_parallel(audio_path, is_url, asr_model, gpu_device)
                        else:
                            # 标准模式：一卡一任务
                            try:
                                # 如果GPU池为空（主服务容器没有GPU），根据全局任务计数器轮询选择GPU服务
                                if total_gpus == 0:
                                    # 从环境变量获取可用的GPU实例数量，默认7个（funasr-gpu0 到 funasr-gpu6）
                                    max_gpu_instances = int(os.getenv("MAX_FUNASR_GPU_INSTANCES", "7"))
                                    # 使用全局任务计数器确保跨请求的负载均衡
                                    global_task_index = await get_next_task_index()
                                    selected_gpu_id = global_task_index % max_gpu_instances
                                    # 构建虚拟的 gpu_device 字符串，用于后续URL构建
                                    gpu_device = f"cuda:{selected_gpu_id}"
                                    logger.info(f"🔄 GPU池为空，根据全局任务索引 {global_task_index} (请求内索引 {idx}) 轮询选择 GPU 实例: {selected_gpu_id}")
                                
                                async with gpu_pool.get_gpu_context() as gpu_device_from_pool:
                                    if gpu_device_from_pool:
                                        gpu_device = gpu_device_from_pool
                                        logger.info(f"🎯 从GPU池获取GPU: {gpu_device}")
                                    elif total_gpus == 0:
                                        # GPU池为空，使用轮询选择的GPU ID
                                        logger.info(f"🎯 使用轮询选择的GPU服务: {gpu_device}")
                                    
                                    result = await handle_audio_parallel(audio_path, is_url, asr_model, gpu_device)
                            except Exception as e:
                                logger.warning(f"⚠️ GPU分配失败，使用默认设备: {e}")
                                result = await handle_audio_parallel(audio_path, is_url, asr_model, None)
                        # 初始化默认值
                        single_raw_text = ""
                        single_transcript_data = []
                        
                        if result and len(result) >= 2:
                            # 提取结果，但确保不为 None
                            single_raw_text = result[0] if result[0] is not None else ""
                            single_transcript_data = result[1] if result[1] is not None else []
                            if len(result) >= 3:
                                asr_task_id = result[2]
                        elif result:
                            # result 存在但长度不足，尝试提取
                            single_raw_text = result[0] if len(result) > 0 and result[0] is not None else ""
                            single_transcript_data = result[1] if len(result) > 1 and result[1] is not None else []
                        
                        # 最终检查：确保 single_raw_text 和 single_transcript_data 不为 None
                        if single_raw_text is None:
                            single_raw_text = ""
                        if single_transcript_data is None:
                            single_transcript_data = []
                    else:
                        # 使用其他ASR服务（如腾讯云）
                        # 确保变量已初始化
                        if 'single_raw_text' not in locals():
                            single_raw_text = ""
                        if 'single_transcript_data' not in locals():
                            single_transcript_data = []
                        
                        try:
                            asr_service = get_asr_service_by_name(asr_model)
                            # 如果使用腾讯云ASR且原始URL存在，使用原始URL而不是本地路径
                            if asr_model == "tencent" and original_url:
                                logger.info(f"🔗 腾讯云ASR使用原始URL: {original_url}")
                                asr_input = original_url
                            else:
                                asr_input = audio_path
                            
                            asr_result = await asyncio.to_thread(asr_service.transcribe, asr_input)
                            if asr_result:
                                single_raw_text = asr_result.get("text", "") or ""
                                single_transcript_data = asr_result.get("transcript", []) or []
                                # 提取流水号
                                asr_task_id = asr_result.get("task_id")
                                if asr_task_id:
                                    asr_task_id = str(asr_task_id)
                                    logger.info(f"📝 ASR服务返回流水号: {asr_task_id}")
                        except Exception as e:
                            logger.warning(f"⚠️ ASR服务调用失败: {e}")
                            # 确保即使失败也有默认值
                            if 'single_raw_text' not in locals() or single_raw_text is None:
                                single_raw_text = ""
                            if 'single_transcript_data' not in locals() or single_transcript_data is None:
                                single_transcript_data = []
                    
                    # 如果没有获取到流水号，生成一个
                    if not asr_task_id:
                        asr_task_id = str(uuid.uuid4())
                        logger.info(f"📝 生成ASR唯一标识: {asr_task_id}")
                    
                    # 声纹匹配（可选）
                    try:
                        from app.services.voice_service import voice_service
                        if voice_service.enabled and single_transcript_data and not is_url:
                            segments = await asyncio.to_thread(
                                voice_service.extract_speaker_segments, audio_path, single_transcript_data
                            )
                            matched = await asyncio.to_thread(voice_service.match_speakers, segments)
                            single_transcript_data = voice_service.replace_speaker_ids(single_transcript_data, matched)
                    except Exception as ve:
                        logger.warning(f"⚠️ 当前音频声纹匹配异常跳过: {ve}")
                    
                    if single_transcript_data:
                        # 优先使用 audio_names 中保存的 audio_id（如果通过JSON格式传入）
                        # 否则使用全局的 audio_id 参数，最后才使用 audio_name
                        current_audio_id = audio_names[idx] if idx < len(audio_names) else None
                        # 检查是否是数字ID（从JSON传入的audio_id通常是数字或数字字符串）
                        if current_audio_id and (isinstance(current_audio_id, (int, str)) and (str(current_audio_id).isdigit() or current_audio_id != audio_name)):
                            # 如果 audio_names 中保存的是业务ID，使用它
                            final_audio_id = current_audio_id
                        elif audio_id is not None:
                            # 使用全局的 audio_id 参数
                            final_audio_id = audio_id
                        else:
                            # 最后使用 audio_name
                            final_audio_id = audio_name
                        
                        for item in single_transcript_data:
                            # 如果时间戳已经是毫秒（大于1000），直接使用；否则认为是秒，需要转换
                            start_raw = item.get("start_time", 0.0)
                            end_raw = item.get("end_time", start_raw)
                            start_sec = float(start_raw) if float(start_raw) < 1000 else float(start_raw) / 1000.0
                            end_sec = float(end_raw) if float(end_raw) < 1000 else float(end_raw) / 1000.0
                            # 加上时间偏移并转换为毫秒
                            item["start_time"] = int((start_sec + time_offset) * 1000)
                            item["end_time"] = int((end_sec + time_offset) * 1000)
                            item["audio_id"] = final_audio_id
                            item["asr_task_id"] = asr_task_id  # 添加 ASR 任务流水号
                        
                        # 计算最后一个片段的结束时间（秒），用于更新 time_offset
                        last_end_time_sec = max((float(item.get("end_time", 0)) / 1000.0) for item in single_transcript_data)
                        
                        # 假设 get_audio_duration 已被正确导入
                        actual_duration = get_audio_duration(audio_path)
                        if actual_duration > 0 and actual_duration > last_end_time_sec - time_offset:
                            time_offset += actual_duration
                        else:
                            time_offset = last_end_time_sec
                    else:
                        actual_duration = get_audio_duration(audio_path)
                        if actual_duration > 0: 
                            time_offset += actual_duration
                    
                    # 最终检查：确保 single_raw_text 不为 None（防御性编程）
                    if single_raw_text is None:
                        logger.warning(f"⚠️ [{idx+1}/{len(audio_paths)}] 音频处理返回 None raw_text，使用空字符串")
                        single_raw_text = ""
                    # 确保是字符串类型
                    if not isinstance(single_raw_text, str):
                        logger.warning(f"⚠️ [{idx+1}/{len(audio_paths)}] raw_text 不是字符串类型: {type(single_raw_text)}，转换为字符串")
                        single_raw_text = str(single_raw_text) if single_raw_text is not None else ""
                    
                    all_raw_texts.append(single_raw_text)
                    # 检查 single_transcript_data 是否为 None，防止 TypeError
                    if single_transcript_data is None:
                        logger.warning(f"⚠️ [{idx+1}/{len(audio_paths)}] 音频处理返回 None，跳过 transcript 数据")
                        single_transcript_data = []
                    all_transcript_data.extend(single_transcript_data)
                    logger.info(f"✅ [{idx+1}/{len(audio_paths)}] 音频处理完成，累计时长: {time_offset:.2f}秒")

                # 过滤掉 None 和空字符串，然后拼接
                valid_texts = [text for text in all_raw_texts if text is not None and text.strip()]
                raw_text = " ".join(valid_texts) if valid_texts else ""
                transcript_data = all_transcript_data

                # 调整 transcript 时间轴为连续，便于前端"点哪播哪"
                if transcript_data:
                    make_transcript_timestamps_continuous(transcript_data)
            
            # 音频处理完成，释放GPU资源，后续LLM调用不受GPU限制
            logger.info("✅ 音频处理完成，释放GPU资源，LLM调用将不受GPU限制并行执行")

        # 基本校验：必须有可用文本（来自ASR或 text_content）
        if not raw_text:
            raise HTTPException(status_code=400, detail="未能提取有效文本内容")

        # --- 历史检索与 LLM 生成（不受GPU限制，可真正并行）---
        history_context = None
        if history_meeting_ids:
            from app.services.meeting_history import meeting_history_service
            m_ids = [i.strip() for i in history_meeting_ids.split(",")]
            if history_mode == "retrieval":
                history_context = await meeting_history_service.process_by_retrieval(m_ids, user_requirement, raw_text, llm_model)
            else:
                history_context = await meeting_history_service.process_by_summary(m_ids, user_requirement, llm_model)

        # --- 1. LLM 初始化 ---
        llm_service = get_llm_service_by_name(llm_model)
        llm_service.temperature, llm_service.max_tokens = llm_temperature, llm_max_tokens
        
        # --- 2. 获取模板配置 ---
        template_config = prompt_template_service.get_template_config(template_id=template)
        
        # --- 3. 选择用于 LLM 的转写文本 / 基础内容 ---
        if use_transcript_for_llm and transcript_data and len(transcript_data) > 0:
            formatted_transcript = format_transcript_for_llm(transcript_data)
            base_transcript_for_llm = formatted_transcript if formatted_transcript else raw_text
            logger.info(f"📝 已启用结构化 transcript 作为 LLM 输入（{len(transcript_data)} 个片段）")
        else:
            base_transcript_for_llm = raw_text
            if use_transcript_for_llm:
                logger.warning("⚠️ use_transcript_for_llm 已开启，但 transcript_data 为空，回退为 raw_text")
            else:
                logger.info("📝 当前使用 raw_text 作为 LLM 输入（默认模式）")

        # --- 3.1 判断用户是否意图“在现有会议纪要基础上修改” ---
        modify_on_existing = False
        if existing_minutes_html and existing_minutes_html.strip() and user_requirement:
            modify_on_existing = await is_modify_minutes_intent(user_requirement)

        if modify_on_existing:
            # 用户显式要求基于现有纪要修改：以 existing_minutes_html 为主输入
            transcript_for_llm = existing_minutes_html
            logger.info("✏️ 检测到用户意图为『基于现有会议纪要进行修改』，优先使用 existing_minutes_html 作为 LLM 主输入")
        else:
            # 否则使用转写/文本内容作为主输入（兼容 rebuild/text_content/音频）
            transcript_for_llm = base_transcript_for_llm

        # --- 4. 使用模板服务渲染最终 Prompt ---
        # 模板内部会注入：用户自定义模板 + 用户特别要求(最高优先级) + 历史会议 + 附件参考文档
        # 如果当前是“仅文本模式”（没有上传音频 / 没有 rebuild），为 LLM 添加一段系统级说明，
        # 避免出现“录音未明确”之类突兀的表述，而是自然说明“信息不明确/目前仅基于文字信息”。
        effective_user_requirement = user_requirement or ""
        if text_only_mode and not rebuild:
            extra_hint = (
                "\n\n【系统说明】当前未提供会议录音或逐字转写，仅根据上方文字信息生成一个概要性会议纪要。"
                "遇到信息缺失或无法确认的部分，请在合适位置自然说明“信息不明确”或“暂无相关内容”，"
                "不要使用“录音未明确”等与录音相关的表述，也不要编造不存在的细节。"
            )
            effective_user_requirement = (effective_user_requirement + extra_hint).strip()
        
        final_prompt = prompt_template_service.render_prompt(
            template_config,
            transcript_for_llm,
            history_context,
            effective_user_requirement,
            reference_document=reference_document_text,
            raw_text=raw_text,
        )
        
        # --- 5. 并行调用 LLM 生成纪要和说话人摘要 ---
        async def generate_meeting_minutes():
            """异步生成会议纪要"""
            try:
                if hasattr(llm_service, "chat"):
                    structured_data = await asyncio.to_thread(
                        llm_service.chat,
                        final_prompt
                    )
                else:
                    structured_data = await asyncio.to_thread(
                        llm_service.generate_markdown,
                        raw_text, "", template, user_requirement
                    )
                # 记录本次生成会议纪要的 token 使用情况（仅此一次调用）
                minutes_usage = getattr(llm_service, "last_usage", None)
                return structured_data, minutes_usage
            except Exception as e:
                logger.error(f"❌ 生成会议纪要失败: {e}")
                raise

        async def generate_speaker_summaries_wrapper():
            """异步生成说话人摘要（包装函数）"""
            if not transcript_data or not any(item.get("speaker_id") is not None for item in transcript_data):
                return None
            try:
                speaker_summaries_dict = await generate_speaker_summaries(transcript_data, llm_service, max_summary_length=None)
                # 将字典转换为数组格式，按照 API_DOCUMENTATION.md 的要求
                speaker_summaries_list = []
                for k, v in speaker_summaries_dict.items():
                    if isinstance(v, dict):
                        speaker_summaries_list.append(v)
                    else:
                        # 如果是 Pydantic 模型，转换为字典
                        speaker_summaries_list.append(
                            v.model_dump() if hasattr(v, "model_dump") else (v.dict() if hasattr(v, "dict") else v)
                        )
                summaries_usage = getattr(llm_service, "last_usage", None)
                return (speaker_summaries_list if speaker_summaries_list else None), summaries_usage
            except Exception as e:
                logger.warning(f"⚠️ 说话人摘要生成过程出错: {e}")
                return None

        # 并行执行两个任务
        logger.info("🚀 开始并行生成会议纪要和说话人摘要...")
        minutes_result, summaries_result = await asyncio.gather(
            generate_meeting_minutes(),
            generate_speaker_summaries_wrapper(),
            return_exceptions=True
        )

        # 处理会议纪要结果
        if isinstance(minutes_result, Exception):
            logger.error(f"❌ 生成会议纪要异常: {minutes_result}")
            raise minutes_result
        
        structured_data, minutes_usage = minutes_result
        clean_md = structured_data.replace("```markdown", "").replace("```", "").strip()
        final_html = markdown.markdown(clean_md, extensions=['nl2br', 'tables'])

        # 如果是重新生成（rebuild 模式），对 HTML 做一次简单清洗，去掉纯空白段落，避免富文本中出现一行文字一行空行的情况
        if rebuild:
            try:
                import re as _re
                # 删除只包含空白或 <br> 的 <p> 段落
                final_html = _re.sub(r"<p>(?:\s|&nbsp;|<br\s*/?>)*</p>", "", final_html, flags=_re.IGNORECASE)
            except Exception as _clean_err:
                logger.warning(f"⚠️ rebuild 模式下清洗 HTML 空段落失败: {_clean_err}")

        # 处理说话人摘要结果
        speaker_summaries = None
        summaries_usage = None
        if isinstance(summaries_result, Exception):
            logger.warning(f"⚠️ 生成说话人摘要异常: {summaries_result}")
        elif summaries_result is not None:
            # 正常情况下 summaries_result 是 (speaker_summaries_list, summaries_usage)
            try:
                speaker_summaries, summaries_usage = summaries_result
            except Exception:
                # 向后兼容：如果返回的不是二元组，尽量按旧格式处理
                speaker_summaries = summaries_result

        # --- 6. 提取 Token 统计 ---
        total_tokens = 0
        input_tokens = 0
        output_tokens = 0
        # 会议纪要 + 说话人摘要 两次 LLM 调用的 token 总和
        for usage_info in (minutes_usage, summaries_usage):
            if isinstance(usage_info, dict):
                total_tokens += int(usage_info.get("total_tokens") or 0)
                input_tokens += int(usage_info.get("prompt_tokens") or 0)
                output_tokens += int(usage_info.get("completion_tokens") or 0)

        # --- 7. 挂载字级别 words（用于前端"点哪播哪"等精细功能） ---
        if transcript_data:
            try:
                # 该函数现在会直接在每个 segment 上生成非重叠、连续的 words 列表：
                # "words": [{"offsetStartMs": ..., "offsetEndMs": ..., "word": "字"}, ...]
                build_word_level_from_transcript(transcript_data)
            except Exception as we:
                logger.warning(f"⚠️ 构建/挂载字级别 words 失败: {we}")

        # --- 9. 返回结果 ---
        return {
            "status": "success",
            "message": "处理成功",
            "raw_text": raw_text,
            "transcript": transcript_data,
            "html_content": final_html,
            "speaker_summaries": speaker_summaries,
            "usage_tokens": total_tokens,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    except Exception as e:
        logger.error(f"❌ 处理异常: {e}\n{traceback.format_exc()}")
        return {"status": "error", "message": str(e), "transcript": []}
    finally:
        try:
            cleanup_files(temp_to_clean)
        except Exception:
            pass


@router.post("/archive", response_model=ArchiveResponse)
async def archive_meeting_knowledge(request: ArchiveRequest):
    """
    归档接口: 接收最终版纪要 -> 切片 -> 向量化 -> 存入 Chroma
    """
    try:
        logger.info(f"💾 收到归档请求: ID={request.minutes_id}, 长度={len(request.markdown_content)}")
        
        if not request.markdown_content.strip():
            return ArchiveResponse(status="failed", message="内容不能为空")

        # 检查向量服务是否可用
        if not vector_service or not vector_service.is_available():
            return ArchiveResponse(
                status="failed", 
                message="向量服务不可用，请检查Chroma配置"
            )

        # 1. 调用向量服务保存数据
        # 这里的 save_knowledge 会自动把长文本切成 500 字的小块
        saved_chunks = vector_service.save_knowledge(
            text=request.markdown_content,
            source_id=request.minutes_id,
            extra_meta={"user_id": request.user_id}
        )

        # 2. 使用实际保存的切片数量
        estimated_chunks = saved_chunks if saved_chunks > 0 else len(request.markdown_content) // 500 + 1

        logger.info(f"✅ 归档成功! ID={request.minutes_id}")
        
        return ArchiveResponse(
            status="success", 
            message="已成功存入企业知识库",
            chunks_count=estimated_chunks
        )

    except Exception as e:
        logger.error(f"❌ 归档失败: {str(e)}")
        # 即使报错也不要让 Java 那边崩溃，返回错误信息即可
        return ArchiveResponse(status="error", message=f"归档异常: {str(e)}")
    

import subprocess
from app.core.config import settings

# logger = logging.getLogger(__name__)
# 假设 router 已经定义
# router = APIRouter()
    
@router.post("/api/voice/register")
async def register_employee_voice(
    file: Optional[UploadFile] = File(None, description="员工录音文件(wav/mp3)，与 audio_url/file_path 三选一"),
    audio_url: Optional[str] = Form(None, description="音频文件URL地址（HTTP/HTTPS），与 file/file_path 三选一"),
    file_path: Optional[str] = Form(None, description="本地音频文件路径，与 file/audio_url 三选一"),
    name: str = Form(..., description="员工姓名"),
    employee_id: str = Form(..., description="员工工号(唯一标识)")
):
    """
    【声纹注册接口】供后端调用
    1. 接收音频流、音频URL或本地文件路径
    2. 转向量
    3. 存入 Chroma
    
    支持三种输入方式（三选一）：
    1. 文件上传：使用 file 参数
    2. URL地址：使用 audio_url 参数
    3. 本地路径：使用 file_path 参数
    """
    overall_start = time.time()
    download_start = 0 
    convert_start = 0
    extract_start = 0
    temp_file_path = None
    # 记录是否需要清理最终的 temp_file_path
    should_cleanup = False
    
    try:
        # 1. 参数验证：file、audio_url 和 file_path 必须三选一
        provided_params = sum([file is not None, audio_url is not None, file_path is not None])
        
        if provided_params == 0:
            return {
                "code": 400,
                "message": "请提供 file（文件上传）、audio_url（URL地址）或 file_path（本地路径）参数之一",
                "data": None
            }
        
        if provided_params > 1:
            return {
                "code": 400,
                "message": "file、audio_url 和 file_path 参数不能同时提供，请只选择一种方式",
                "data": None
            }
        
        # 确保目录存在
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # 2. 处理文件上传
        if file:
            file_ext = os.path.splitext(file.filename)[1] or ".wav"
            temp_filename = f"reg_{employee_id}_{uuid.uuid4()}{file_ext}"
            temp_file_path = settings.TEMP_DIR / temp_filename
            should_cleanup = True
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"📥 收到注册请求（文件上传）: {name} (工号:{employee_id})")
        
        # 3. 处理URL地址
        elif audio_url:
            download_start = time.time()
            if not audio_url.startswith(("http://", "https://")):
                return {
                    "code": 400,
                    "message": "audio_url 必须是有效的 HTTP/HTTPS URL",
                    "data": None
                }
            
            try:
                logger.info(f"📥 收到注册请求（URL）: {name} (工号:{employee_id}), URL: {audio_url}")
                logger.info(f"🔗 开始下载音频: {audio_url}")
                
                # 下载文件
                response = requests.get(audio_url, stream=True, timeout=300)
                response.raise_for_status()
                
                # 从URL或Content-Disposition获取文件名
                filename = os.path.basename(audio_url)
                content_disposition = response.headers.get('Content-Disposition', '')
                if 'filename=' in content_disposition:
                    import re
                    match = re.search(r'filename=["\']?([^"\']+)["\']?', content_disposition)
                    if match:
                        filename = match.group(1)
                
                # 如果没有扩展名，尝试从Content-Type推断
                if not os.path.splitext(filename)[1]:
                    content_type = response.headers.get('Content-Type', '')
                    if 'audio' in content_type.lower():
                        if 'wav' in content_type.lower(): filename += '.wav'
                        elif 'mp3' in content_type.lower(): filename += '.mp3'
                        elif 'm4a' in content_type.lower(): filename += '.m4a'
                        else: filename += '.wav'  # 默认 wav
                
                file_ext = os.path.splitext(filename)[1] or ".wav"
                temp_filename = f"reg_{employee_id}_{uuid.uuid4()}{file_ext}"
                temp_file_path = settings.TEMP_DIR / temp_filename
                should_cleanup = True
                
                # 流式写入文件
                with open(temp_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
                logger.info(f"✅ 下载完成: {filename} ({file_size_mb:.2f} MB)")
                download_duration = time.time() - download_start
                logger.info(f"⏱️ [阶段1-下载耗时]: {download_duration:.2f}秒") 
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ 下载失败: {e}")
                return {
                    "code": 400,
                    "message": f"音频文件下载失败: {str(e)}",
                    "data": None
                }
        
        # 4. 处理本地文件路径
        elif file_path:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {
                    "code": 400,
                    "message": f"音频文件不存在: {file_path}",
                    "data": None
                }
            # 直接使用本地路径，原生路径不清理
            temp_file_path = file_path
            should_cleanup = False
            logger.info(f"📥 收到注册请求（本地路径）: {name} (工号:{employee_id}), 路径: {file_path}")
        
        # 5. 检查音频文件质量和格式（强制格式转换与长度校验）
        try:
            import soundfile as sf
            
            path_str = str(temp_file_path)
            convert_start = time.time()
            
            if not os.path.exists(path_str):
                return {"code": 400, "message": "音频文件保存失败，请重试", "data": None}
            
            # 核心防御：如果不是 wav 格式，强制先用 ffmpeg 转成 wav
            if not path_str.lower().endswith(".wav"):
                logger.info(f"🔄 检测到非 wav 格式，正在强制转换以校验长度: {path_str}")
                wav_path = str(settings.TEMP_DIR / f"reg_converted_{uuid.uuid4().hex}.wav")
                
                # 调用 ffmpeg 转换，统一采样率为 16000Hz 单声道
                cmd = ["ffmpeg", "-i", path_str, "-ac", "1", "-ar", "16000", "-y", wav_path]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 如果原始文件是刚才上传/下载的临时文件，清理它
                if should_cleanup and os.path.exists(path_str):
                    os.remove(path_str)
                
                # 将指针指向新生成的文件，并标记需要清理
                temp_file_path = wav_path
                path_str = wav_path
                should_cleanup = True

            convert_duration = time.time() - convert_start
            logger.info(f"⏱️ [阶段2-转换耗时]: {convert_duration:.2f}秒")

            info = sf.info(path_str)
            duration = info.frames / info.samplerate
            sample_rate = info.samplerate
            
            logger.info(f"📊 注册音频信息: 时长={duration:.2f}秒, 采样率={sample_rate}Hz, 格式={info.format}")
            
            # 严格防线：只允许 2秒 ~ 60秒 之间的录音注册
            if duration < 2.0:
                return {"code": 400, "message": f"音频过短（{duration:.2f}秒），至少需要 2 秒"}
            if duration > 60.0:
                return {"code": 400, "message": f"音频过长（{duration:.2f}秒），声纹注册请上传 60 秒以内的个人纯净语音！"}
                
        except Exception as e:
            logger.error(f"❌ 音频校验/转换失败: {e}")
            return {"code": 400, "message": f"音频格式不受支持或转换失败。错误: {str(e)}", "data": None}
        
        # 6. 延迟导入 voice_service
        try:
            from app.services.voice_service import voice_service
        except ImportError as e:
            logger.error(f"❌ 声纹服务未安装或依赖缺失: {e}")
            return {"code": 500, "message": "声纹服务未安装，请联系管理员", "data": None}
        
        # 7. 调用服务提取向量
        extract_start = time.time()
        vector = voice_service.extract_vector(str(temp_file_path))
        extract_duration = time.time() - extract_start
        logger.info(f"⏱️ [阶段3-算法提取耗时]: {extract_duration:.2f}秒")
        
        if not vector:
            return {"code": 400, "message": "无法提取声纹特征，可能原因：音频质量差或包含多个人声", "data": None}

        # 8. 存入库
        voice_service.save_identity(employee_id, name, vector)

        total_cost = time.time() - overall_start
        logger.info(f"🚀 [注册流程完成] 总耗时: {total_cost:.2f}秒") 
        return {
            "code": 200,
            "message": f"注册成功,处理总耗时 {total_cost:.2f}s",
            "data": {
                "employee_id": employee_id,
                "name": name,
                "vector_dim": len(vector)
            }
        }

    except Exception as e:
        logger.error(f"注册接口异常: {e}")
        return {"code": 500, "message": f"服务端内部错误: {str(e)}"}
        
    finally:
        # 9. 清理临时文件
        if temp_file_path and should_cleanup:
            path_str = str(temp_file_path)
            if os.path.exists(path_str):
                try:
                    os.remove(path_str)
                    logger.info(f"🧹 已安全清理临时文件: {path_str}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理临时文件失败: {e}")
# =============================================
# 热词管理接口
# =============================================

@router.get("/api/hotwords")
async def get_hotwords():
    """
    【获取热词列表】
    转发到FunASR服务获取热词
    """
    try:
        import requests
        from app.core.config import settings
        
        # 构建FunASR服务URL
        funasr_url = getattr(settings, "FUNASR_SERVICE_URL", "http://localhost:8002")
        response = requests.get(f"{funasr_url}/hotwords", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            return {
                "code": 200,
                "message": "获取成功",
                "data": data.get("data", {})
            }
        else:
            return {"code": 500, "message": "FunASR服务返回错误"}
            
    except Exception as e:
        logger.error(f"❌ 获取热词失败: {e}")
        return {"code": 500, "message": f"获取失败: {str(e)}"}


@router.post("/api/hotwords/reload")
async def reload_hotwords():
    """
    【重新加载热词】
    转发到FunASR服务重新加载热词（用于修改funasr_standalone/hotwords.json后刷新）
    """
    try:
        import requests
        from app.core.config import settings
        
        # 构建FunASR服务URL
        funasr_url = getattr(settings, "FUNASR_SERVICE_URL", "http://localhost:8002")
        response = requests.post(f"{funasr_url}/hotwords/reload", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("code") == 0:
                return {
                    "code": 200,
                    "message": "热词重载成功",
                    "data": data.get("data", {})
                }
            else:
                return {"code": 500, "message": data.get("msg", "重载失败")}
        else:
            return {"code": 500, "message": "FunASR服务返回错误"}
            
    except Exception as e:
        logger.error(f"❌ 重载热词失败: {e}")
        return {"code": 500, "message": f"重载失败: {str(e)}"}


# =============================================
# 文档解析接口（仅返回带格式的 HTML）
# =============================================

@router.post("/api/document/parse")
async def parse_document_template(
    file: Optional[UploadFile] = File(None, description="模板文件（支持 .docx, .pdf, .txt）"),
    file_url: Optional[str] = Form(None, description="模板文件URL地址（与file二选一）")
):
    """
    【文档解析接口】
    仅解析模板文件，返回带格式的 HTML5 内容（不进行音频处理或 LLM 生成）
    
    支持两种输入方式（二选一）：
    1. 文件上传：使用 file 参数
    2. URL地址：使用 file_url 参数
    
    支持格式：
    - .docx: 使用 mammoth 转换为语义化 HTML
    - .pdf: 优先使用 pdf2htmlEX 高保真还原，否则退化为文本 HTML
    - .txt: 简单包装为 <pre> 格式的 HTML
    """
    temp_file_path = None
    filename = None
    
    try:
        # 1. 参数验证：file 和 file_url 必须二选一
        if not file and not file_url:
            return {
                "code": 400,
                "message": "请提供 file（文件上传）或 file_url（URL地址）参数之一",
                "data": None
            }
        
        if file and file_url:
            return {
                "code": 400,
                "message": "file 和 file_url 参数不能同时提供，请只选择一种方式",
                "data": None
            }
        
        # 确保目录存在
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        # 2. 处理文件上传
        if file:
            file_ext = os.path.splitext(file.filename)[1] or ""
            temp_filename = f"template_{uuid.uuid4().hex}{file_ext}"
            temp_file_path = settings.TEMP_DIR / temp_filename
            filename = file.filename
            
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            logger.info(f"📄 收到文档解析请求（文件上传）: {filename}")
        
        # 3. 处理URL地址
        elif file_url:
            if not file_url.startswith(("http://", "https://")):
                return {
                    "code": 400,
                    "message": "file_url 必须是有效的 HTTP/HTTPS URL",
                    "data": None
                }
            
            try:
                logger.info(f"📥 开始下载文档: {file_url}")
                
                # 下载文件
                response = requests.get(file_url, stream=True, timeout=300)
                response.raise_for_status()
                
                # 从URL或Content-Disposition获取文件名
                filename = os.path.basename(file_url)
                content_disposition = response.headers.get('Content-Disposition', '')
                if 'filename=' in content_disposition:
                    import re
                    match = re.search(r'filename=["\']?([^"\']+)["\']?', content_disposition)
                    if match:
                        filename = match.group(1)
                
                # 如果没有扩展名，尝试从Content-Type推断
                if not os.path.splitext(filename)[1]:
                    content_type = response.headers.get('Content-Type', '')
                    if 'pdf' in content_type.lower():
                        filename += '.pdf'
                    elif 'word' in content_type.lower() or 'document' in content_type.lower():
                        filename += '.docx'
                    elif 'text' in content_type.lower():
                        filename += '.txt'
                
                file_ext = os.path.splitext(filename)[1] or ""
                temp_filename = f"template_{uuid.uuid4().hex}{file_ext}"
                temp_file_path = settings.TEMP_DIR / temp_filename
                
                # 流式写入文件
                with open(temp_file_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                file_size_mb = os.path.getsize(temp_file_path) / (1024 * 1024)
                logger.info(f"✅ 下载完成: {filename} ({file_size_mb:.2f} MB)")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ 下载失败: {e}")
                return {
                    "code": 400,
                    "message": f"文件下载失败: {str(e)}",
                    "data": None
                }
        
        # 4. 调用文档服务提取 HTML（带格式）
        html_content = document_service.extract_html_from_file(str(temp_file_path))
        
        if not html_content:
            return {
                "code": 400,
                "message": "文档解析失败，请检查文件格式是否正确（支持 .docx, .pdf, .txt）",
                "data": None
            }
        
        logger.info(f"✅ 文档解析成功，HTML 长度: {len(html_content)}")
        
        return {
            "code": 200,
            "message": "解析成功",
            "data": {
                "filename": filename,
                "html_content": html_content
            }
        }
        
    except Exception as e:
        logger.error(f"❌ 文档解析接口异常: {e}\n{traceback.format_exc()}")
        return {
            "code": 500,
            "message": f"服务端内部错误: {str(e)}",
            "data": None
        }
        
    finally:
        # 5. 清理临时文件
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(f"🧹 已清理临时文件: {temp_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ 清理临时文件失败: {e}")