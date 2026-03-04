import os, shutil, uuid, tempfile, markdown, requests, traceback
from typing import Optional, List, Dict
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

async def generate_speaker_summaries(
    transcript_data: List[Dict], 
    llm_service, 
    max_summary_length: int = None
) -> Dict[str, SpeakerSummary]:
    if not transcript_data:
        return {}
    
    # 1. 按说话人分组收集文本 (这部分逻辑保持不变)
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

    logger.info(f"📊 开始为 {len(speaker_texts)} 个说话人并发生成摘要...")

    # 💡 内部异步子任务：处理单个说话人的总结
    async def process_single_speaker(speaker_key, data):
        full_text = " ".join(data["texts"])
        word_count = len(full_text)
        
        if word_count < 500:
            input_limit, summary_length_hint, max_tokens = word_count, "100-200字", 500
        elif word_count < 2000:
            input_limit, summary_length_hint, max_tokens = min(word_count, 5000), "200-400字", 1000
        elif word_count < 5000:
            input_limit, summary_length_hint, max_tokens = min(word_count, 8000), "400-800字", 2000
        else:
            input_limit, summary_length_hint, max_tokens = min(word_count, 12000), "800-1500字", 3000
            
        limited_text = full_text[:input_limit]
        speaker_display_name = data.get("name") or speaker_key
        
        if max_summary_length:
            length_instruction = f"控制在{max_summary_length}字以内"
        else:
            length_instruction = f"建议控制在{summary_length_hint}，但可以根据内容重要性适当扩展，确保关键信息完整"

        # 优化后的严谨 Prompt
        prompt = f"""你是一位拥有10年经验的企业高级高管助理兼数据架构专家。请为以下会议发言内容生成详细的专业摘要，要求：
1. 核心总结：提取该说话人的主要观点、业务痛点和后续计划。
2. 决策聚焦：突出关键信息、核心数据和决策点。
3. 篇幅要求：{length_instruction}。
4. 【红线约束1：静默容错与自动修正】：原始文本由机器识别，包含大量同音错别字、口语化和逻辑跳跃。请你凭借专业知识，在心中自行脑补、修正并翻译成专业的商业/技术术语。绝对禁止在输出结果中提及“文本错误”、“逻辑混乱”或“表述不清”。
5. 【红线约束2：严禁主观评价】：你的唯一职责是“提炼业务价值”。绝对禁止对发言人的身份、语言表达能力、逻辑性进行任何评价、质疑或猜测。禁止输出任何类似“身份存疑”、“矛盾点”等评价性字眼。
6. 【红线约束3：输出格式纯净】：直接输出客观、中立、精炼的结构化会议纪要，不要包含任何对工作过程的解释或免责声明。

【说话人】：{speaker_display_name}
【发言内容】：
{limited_text}

【专业摘要】："""

        try:
            # 🚀 核心优化：使用 to_thread 将同步请求抛入后台线程池，绝不阻塞 FastAPI 主循环！
            response_text = await asyncio.to_thread(
                llm_service.chat,
                prompt, 
                temperature=0.3, 
                max_tokens=max_tokens
            )
            
            # 防御性编程：防止 LLM 抽风返回 None 导致 .strip() 崩溃
            summary_text = (response_text or "").strip()
            if not summary_text:
                summary_text = "暂无有效摘要内容。"
                
            if max_summary_length and len(summary_text) > max_summary_length:
                summary_text = summary_text[:max_summary_length] + "..."
                
        except Exception as e:
            logger.warning(f"⚠️ 为说话人 {speaker_key} 生成摘要失败: {e}")
            if max_summary_length:
                summary_text = full_text[:max_summary_length] + "..." if len(full_text) > max_summary_length else full_text
            else:
                summary_text = full_text[:1000] + "..." if len(full_text) > 1000 else full_text

        logger.info(f"✅ 说话人 {speaker_key} ({speaker_display_name}) 摘要生成完成: {len(summary_text)} 字")
        
        return speaker_key, SpeakerSummary(
            speaker_id=speaker_key,
            speaker_name=data.get("name"),
            summary=summary_text,
            word_count=word_count,
            speech_segments=data["segments"]
        )

    # 2. 进阶版：使用 Semaphore 限制并发数量
    import asyncio
    
    # 设定最大并发数，建议设为 2（根据大模型 API 额度，如果还报 429 就降为 1）
    max_concurrent_requests = 2
    sem = asyncio.Semaphore(max_concurrent_requests)

    # 创建一个包装器，将原有的处理函数套在“收费站”里面
    async def bounded_process(speaker_key, data):
        async with sem:
            # 拿到通行证后，在真正发请求前稍微错开 1.5 秒
            # 防止即使是 2 个请求也刚好在同一毫秒发出导致瞬间 TPM 尖峰
            await asyncio.sleep(1.5)
            return await process_single_speaker(speaker_key, data)

    logger.info(f"🚦 启用并发控制：最大同时处理 {max_concurrent_requests} 个请求...")
    
    # 生成带有并发限制的任务列表
    tasks = [bounded_process(key, data) for key, data in speaker_texts.items()]
    
    # gather 依然会接收所有任务，但内部被 Semaphore 卡住，严格按照设定的通道数放行
    results = await asyncio.gather(*tasks)
    
    # 组装返回结果
    summaries = {speaker_key: summary for speaker_key, summary in results}
    
    logger.info(f"✅ 所有说话人摘要带限流并发生成完成，共 {len(summaries)} 个")
    return summaries

async def handle_audio_parallel(audio_path: str, is_url: bool, asr_model: str):
    """封装并行处理逻辑 (全异步流式 I/O，拯救物理内存)"""
    from app.services.parallel_processor import map_words_to_speakers, aggregate_by_speaker, parse_rttm
    funasr_url = os.getenv("FUNASR_SERVICE_URL", "")
    pyannote_url = os.getenv("PYANNOTE_SERVICE_URL", "")

    async def run_funasr():
        url = f"{funasr_url}/transcribe/word-level"
        params = {"hotword": ""}
        # 使用 httpx 进行异步请求，设置超时时间
        async with httpx.AsyncClient(timeout=600.0) as client:
            if is_url:
                resp = await client.post(url, data={"audio_url": audio_path, **params})
                return resp.json().get("words", [])
            else:
                # 异步流式上传文件，内存占用极小
                with open(audio_path, "rb") as f:
                    resp = await client.post(url, files={"file": f}, data=params)
                    return resp.json().get("words", [])

    async def run_pyannote():
        url = f"{pyannote_url}/rttm"
        async with httpx.AsyncClient(timeout=600.0) as client:
            if is_url:
                resp = await client.post(url, data={"audio_url": audio_path})
                return resp.json().get("rttm", "")
            else:
                with open(audio_path, "rb") as f:
                    resp = await client.post(url, files={"file": f})
                    return resp.json().get("rttm", "")

    # 🚀 核心优化：使用 asyncio.gather 替代 ThreadPoolExecutor
    # 彻底释放 CPU 线程，实现真正的并发，极大降低内存占用
    try:
        f_words, f_rttm = await asyncio.gather(run_funasr(), run_pyannote())
    except Exception as e:
        logger.error(f"❌ 并行请求服务失败: {e}")
        return None, None

    if not f_words or not f_rttm:
        logger.warning("⚠️ ASR 或 Pyannote 服务返回为空")
        return None, None

    rttm_segments = parse_rttm(f_rttm)
    mapped = map_words_to_speakers(f_words, rttm_segments)
    transcript_data = aggregate_by_speaker(mapped)

    # 调试信息保留
    try:
        sample_speakers = [item.get("speaker_id") for item in transcript_data[:10]]
        logger.info(f"🔎 并行聚合后原始 speaker_id 样例: {sample_speakers}")
    except Exception as e:
        logger.debug(f"调试 speaker_id 样例失败: {e}")
    
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
        # 去除所有常见的全角/半角标点符号和空白字符
        clean_text = re.sub(r'[^\w\u4e00-\u9fa5a-zA-Z0-9]', '', text)
        return len(clean_text) > 0

    formatted_data = []
    for item in transcript_data:
        text = item.get("text", "").strip()
        
        if not is_valid_text(text):
            logger.debug(f"🗑️ 过滤掉无效的噪音片段: [{item.get('start')} - {item.get('end')}]")
            continue
            
        formatted_data.append({
            "text": text,
            "start_time": item.get("start", 0.0),
            "end_time": item.get("end", 0.0),
            "speaker_id": speaker_str_to_int(item.get("speaker_id", "SPEAKER_00"))
        })
    
    return "".join([i["text"] for i in formatted_data]), formatted_data

# --- 主接口 ---

MAX_CONCURRENT_PROCESS = None

@router.post("/process", response_model=MeetingResponse)
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
    llm_temperature: float = Form(0.7),
    llm_max_tokens: int = Form(2000),
    asr_model: str = Form("auto"),
):
    global MAX_CONCURRENT_PROCESS
    if MAX_CONCURRENT_PROCESS is None:
        MAX_CONCURRENT_PROCESS = asyncio.Semaphore(1)

    temp_to_clean = []
    raw_text, transcript_data = "", []

    try:
        reference_document_text = None
        
        async with MAX_CONCURRENT_PROCESS:
            logger.info("🚦 获取到处理令牌，开始执行任务...")
            
            # 1. 处理文档附件URL
            if document_urls:
                url_list = [url.strip().strip('"') for url in document_urls.split(',') if url.strip()]
                logger.info(f"📎 收到 {len(url_list)} 个文档URL")
                all_document_texts = []
                for idx, doc_url in enumerate(url_list):
                    if not doc_url.startswith(("http://", "https://")): continue
                    try:
                        logger.info(f"📥 [{idx+1}/{len(url_list)}] 开始下载文档: {doc_url}")
                        async with httpx.AsyncClient() as client:
                            async with client.stream("GET", doc_url, timeout=300) as resp:
                                resp.raise_for_status()
                                filename = os.path.basename(doc_url)
                                content_disposition = resp.headers.get('Content-Disposition', '')
                                if 'filename=' in content_disposition:
                                    import re
                                    match = re.search(r'filename=["\']?([^"\']+)["\']?', content_disposition)
                                    if match: filename = match.group(1)
                                
                                if not os.path.splitext(filename)[1]:
                                    content_type = resp.headers.get('Content-Type', '')
                                    if 'word' in content_type.lower() or 'document' in content_type.lower(): filename += '.docx'
                                    elif 'pdf' in content_type.lower(): filename += '.pdf'
                                    elif 'text' in content_type.lower(): filename += '.txt'
                                    else: filename += '.docx'
                                
                                doc_path = settings.TEMP_DIR / f"doc_{uuid.uuid4().hex}_{filename}"
                                with open(doc_path, "wb") as f:
                                    async for chunk in resp.aiter_bytes(chunk_size=8192):
                                        f.write(chunk)
                                        
                        temp_to_clean.append(str(doc_path))
                        logger.info(f"✅ [{idx+1}/{len(url_list)}] 文档下载完成: {filename}")
                        
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

            # 2. 输入源解析与预处理
            if text_content:
                raw_text = text_content
                transcript_data = []
            else:
                audio_paths, is_url_list, audio_names = [], [], []
                
                if files:
                    logger.info(f"📁 收到 {len(files)} 个音频文件上传")
                    for idx, f in enumerate(files):
                        p = settings.TEMP_DIR / f"multi_{uuid.uuid4().hex}_{idx}_{f.filename}"
                        with open(p, "wb") as b: shutil.copyfileobj(f.file, b)
                        temp_to_clean.append(str(p))
                        audio_paths.append(str(p))
                        is_url_list.append(False)
                        audio_names.append(f.filename or f"file_{idx+1}")
                        
                elif audio_urls:
                    url_list = [url.strip().strip('"') for url in audio_urls.split(',') if url.strip()]
                    for idx, raw_url in enumerate(url_list):
                        if raw_url.startswith("http"):
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
                                audio_names.append(base_name)
                            except Exception as e:
                                raise HTTPException(status_code=400, detail=f"无法下载音频URL: {raw_url}, 错误: {str(e)}")
                        else:
                            audio_paths.append(raw_url)
                            is_url_list.append(False)
                            audio_names.append(os.path.basename(raw_url) or raw_url)
                            
                elif file_paths:
                    path_list = [path.strip() for path in file_paths.split(',') if path.strip()]
                    audio_paths.extend(path_list)
                    is_url_list.extend([False] * len(path_list))
                    audio_names.extend([os.path.basename(p) or p for p in path_list])

                if not audio_paths:
                    raise HTTPException(status_code=400, detail="未提供有效的音频")

                logger.info(f"🎵 开始处理 {len(audio_paths)} 个音频...")
                all_raw_texts, all_transcript_data, time_offset = [], [], 0.0
                funasr_url = os.getenv("FUNASR_SERVICE_URL")
                pyannote_url = os.getenv("PYANNOTE_SERVICE_URL")

                for idx, audio_path in enumerate(audio_paths):
                    is_url = is_url_list[idx] if idx < len(is_url_list) else False
                    audio_name = audio_names[idx] if idx < len(audio_names) else os.path.basename(audio_path)
                    logger.info(f"🎤 [{idx+1}/{len(audio_paths)}] 处理音频: {audio_name} ({audio_path})")
                    
                    single_raw_text = ""
                    single_transcript_data = []
                    
                    if funasr_url and pyannote_url and asr_model == "funasr":
                        logger.info(f"🚀 启动并行处理引擎...")
                        single_raw_text, single_transcript_data = await handle_audio_parallel(audio_path, is_url, asr_model)
                    
                    try:
                        from app.services.voice_service import voice_service
                        if voice_service.enabled and single_transcript_data and not is_url:
                            segments = await asyncio.to_thread(voice_service.extract_speaker_segments, audio_path, single_transcript_data)
                            matched = await asyncio.to_thread(voice_service.match_speakers, segments)
                            single_transcript_data = voice_service.replace_speaker_ids(single_transcript_data, matched)
                    except Exception as ve:
                        logger.warning(f"⚠️ 当前音频声纹匹配异常跳过: {ve}")
                    
                    if single_transcript_data:
                        for item in single_transcript_data:
                            item["start_time"] = round(item.get("start_time", 0.0) + time_offset, 2)
                            item["end_time"] = round(item.get("end_time", 0.0) + time_offset, 2)
                            item["audio_name"] = audio_name
                        
                        last_end_time = max(item.get("end_time", 0.0) for item in single_transcript_data)
                        actual_duration = get_audio_duration(audio_path)
                        if actual_duration > 0 and actual_duration > last_end_time - time_offset:
                            time_offset += actual_duration
                        else:
                            time_offset = last_end_time
                    else:
                        actual_duration = get_audio_duration(audio_path)
                        if actual_duration > 0: time_offset += actual_duration
                    
                    all_raw_texts.append(single_raw_text)
                    all_transcript_data.extend(single_transcript_data)
                    logger.info(f"✅ [{idx+1}/{len(audio_paths)}] 音频处理完成，累计时长: {time_offset:.2f}秒")

                raw_text = " ".join(all_raw_texts)
                transcript_data = all_transcript_data

            if not raw_text:
                raise HTTPException(status_code=400, detail="未能提取有效文本内容")

            # 4. 历史检索与 LLM 生成
            history_context = None
            if history_meeting_ids:
                from app.services.meeting_history import meeting_history_service
                m_ids = [i.strip() for i in history_meeting_ids.split(",")]
                if history_mode == "retrieval":
                    history_context = await meeting_history_service.process_by_retrieval(m_ids, user_requirement, raw_text, llm_model)
                else:
                    history_context = await meeting_history_service.process_by_summary(m_ids, user_requirement, llm_model)

            llm_service = get_llm_service_by_name(llm_model)
            llm_service.temperature, llm_service.max_tokens = llm_temperature, llm_max_tokens
            template_config = prompt_template_service.get_template_config(template_id=template)
            
            if transcript_data and len(transcript_data) > 0:
                formatted_transcript = format_transcript_for_llm(transcript_data)
                transcript_for_llm = formatted_transcript if formatted_transcript else raw_text
            else:
                transcript_for_llm = raw_text
            
            final_prompt = prompt_template_service.render_prompt(
                template_config, transcript_for_llm, history_context, user_requirement,
                reference_document=reference_document_text, raw_text=raw_text 
            )
            
            structured_data = llm_service.chat(final_prompt) if hasattr(llm_service, 'chat') else llm_service.generate_markdown(raw_text, "", template, user_requirement)
            clean_md = structured_data.replace("```markdown", "").replace("```", "").strip()
            final_html = markdown.markdown(clean_md, extensions=['nl2br', 'tables'])

            # 5. 生成说话人摘要
            speaker_summaries = None
            if transcript_data and any(item.get("speaker_id") is not None for item in transcript_data):
                try:
                    speaker_summaries_dict = await generate_speaker_summaries(transcript_data, llm_service, max_summary_length=None)
                    speaker_summaries = {k: v.model_dump() if hasattr(v, 'model_dump') else v.dict() for k, v in speaker_summaries_dict.items()}
                except Exception as e:
                    speaker_summaries = None

            return MeetingResponse(
                status="success", message="处理成功", raw_text=raw_text,
                transcript=[TranscriptItem(**item) for item in transcript_data],
                html_content=final_html, speaker_summaries=speaker_summaries
            )

    except Exception as e:
        logger.error(f"❌ 处理异常: {e}\n{traceback.format_exc()}")
        return MeetingResponse(status="error", message=str(e), transcript=[])
    finally:
        cleanup_files(temp_to_clean)


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
                        if 'wav' in content_type.lower():
                            filename += '.wav'
                        elif 'mp3' in content_type.lower():
                            filename += '.mp3'
                        elif 'm4a' in content_type.lower():
                            filename += '.m4a'
                        else:
                            filename += '.wav'  # 默认 wav
                
                file_ext = os.path.splitext(filename)[1] or ".wav"
                temp_filename = f"reg_{employee_id}_{uuid.uuid4()}{file_ext}"
                temp_file_path = settings.TEMP_DIR / temp_filename
                
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
        
        # 3. 处理本地文件路径
        elif file_path:
            # 检查文件是否存在
            if not os.path.exists(file_path):
                return {
                    "code": 400,
                    "message": f"音频文件不存在: {file_path}",
                    "data": None
                }
            
            # 直接使用本地路径
            temp_file_path = file_path
            logger.info(f"📥 收到注册请求（本地路径）: {name} (工号:{employee_id}), 路径: {file_path}")
        
        # 4. 检查音频文件质量和格式（强制格式转换与长度校验）
        try:
            import soundfile as sf
            import subprocess
            convert_start = time.time()
            
            path_str = str(temp_file_path)

            convert_start = time.time()
            
            # 检查文件是否存在
            if not os.path.exists(path_str):
                return {"code": 400, "message": "音频文件保存失败，请重试", "data": None}
            
            # 核心防御：如果不是 wav 格式，强制先用 ffmpeg 转成 wav
            if not path_str.lower().endswith(".wav"):
                logger.info(f"🔄 检测到非 wav 格式，正在强制转换以校验长度: {path_str}")
                wav_path = f"{path_str}_converted.wav"
                
                # 调用 ffmpeg 转换，统一采样率为 16000Hz 单声道
                cmd = ["ffmpeg", "-i", path_str, "-ac", "1", "-ar", "16000", "-y", wav_path]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                
                # 转换成功后，立刻删除旧格式的原文件
                os.remove(path_str)
                # 将路径指向新生成的 wav 文件，让后面的流程使用
                temp_file_path = wav_path
                path_str = wav_path

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
        
        # 3. 延迟导入 voice_service
        try:
            from app.services.voice_service import voice_service
        except ImportError as e:
            logger.error(f"❌ 声纹服务未安装或依赖缺失: {e}")
            return {"code": 500, "message": "声纹服务未安装，请联系管理员", "data": None}
        
        # 6. 调用服务提取向量
        extract_start = time.time()

        vector = voice_service.extract_vector(str(temp_file_path))

        extract_duration = time.time() - extract_start
        logger.info(f"⏱️ [阶段3-算法提取耗时]: {extract_duration:.2f}秒")
        
        if not vector:
            return {"code": 400, "message": "无法提取声纹特征，可能原因：音频质量差或包含多个人声", "data": None}

        # 7. 存入库
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
        # 8. 清理临时文件
        if temp_file_path:
            path_str = str(temp_file_path)  # 强制转为字符串
            if os.path.exists(path_str):
                try:
                    temp_dir_str = str(settings.TEMP_DIR)
                    if path_str.startswith(temp_dir_str):
                        os.remove(path_str)
                        logger.info(f"🧹 已安全清理临时文件: {path_str}")
                    else:
                        logger.info(f"ℹ️ 跳过清理用户提供的本地文件: {path_str}")
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