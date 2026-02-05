import os, shutil, uuid, tempfile, markdown, requests, traceback
from typing import Optional, List
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, UploadFile, File, Form, HTTPException

from app.core.config import settings
from app.core.logger import logger
from app.schemas.task import MeetingResponse, ArchiveRequest, ArchiveResponse, TranscriptItem
from app.services.vector import vector_service
from app.services.asr_factory import get_asr_service_by_name
from app.services.llm_factory import get_llm_service_by_name
from app.services.document import document_service 
from app.services.prompt_template import prompt_template_service

router = APIRouter()

# --- 辅助工具函数 ---

def cleanup_files(files: list):
    """统一清理临时文件"""
    for f in files:
        if f and os.path.exists(f):
            try:
                os.remove(f)
                logger.info(f"🧹 已清理临时文件: {f}")
            except Exception as e:
                logger.warning(f"⚠️ 清理失败 {f}: {e}")

async def handle_audio_parallel(audio_path: str, is_url: bool, asr_model: str):
    """封装并行处理逻辑 (FunASR + Pyannote)"""
    from app.services.parallel_processor import map_words_to_speakers, aggregate_by_speaker, parse_rttm
    funasr_url = os.getenv("FUNASR_SERVICE_URL", "")
    pyannote_url = os.getenv("PYANNOTE_SERVICE_URL", "")

    def run_funasr():
        url = f"{funasr_url}/transcribe/word-level"
        params = {"hotword": ""}
        if is_url:
            return requests.post(url, data={"audio_url": audio_path, **params}, timeout=600).json().get("words", [])
        with open(audio_path, "rb") as f:
            return requests.post(url, files={"file": f}, data=params, timeout=600).json().get("words", [])

    def run_pyannote():
        url = f"{pyannote_url}/rttm"
        if is_url:
            return requests.post(url, data={"audio_url": audio_path}, timeout=600).json().get("rttm", "")
        with open(audio_path, "rb") as f:
            return requests.post(url, files={"file": f}, timeout=600).json().get("rttm", "")

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_words = executor.submit(run_funasr)
        f_rttm = executor.submit(run_pyannote)
        words, rttm_content = f_words.result(), f_rttm.result()

    if not words or not rttm_content:
        return None, None

    rttm_segments = parse_rttm(rttm_content)
    mapped = map_words_to_speakers(words, rttm_segments)
    transcript_data = aggregate_by_speaker(mapped)

    # 调试：看一眼并行聚合后的原始 speaker_id（字符串形式）
    try:
        sample_speakers = [item.get("speaker_id") for item in transcript_data[:10]]
        logger.info(f"🔎 并行聚合后原始 speaker_id 样例: {sample_speakers}")
    except Exception as e:
        logger.debug(f"调试 speaker_id 样例失败: {e}")
    
    # 辅助函数：将 SPEAKER_XX 转换为整数
    def speaker_str_to_int(speaker_str: str) -> Optional[int]:
        """将 'SPEAKER_01' 转换为 1，'SPEAKER_00' 转换为 0"""
        if not speaker_str:
            return None
        try:
            # 提取数字部分
            if isinstance(speaker_str, int):
                return speaker_str
            if isinstance(speaker_str, str) and speaker_str.startswith("SPEAKER_"):
                num_str = speaker_str.replace("SPEAKER_", "").strip()
                return int(num_str) if num_str.isdigit() else None
            # 尝试直接转换
            return int(speaker_str) if str(speaker_str).isdigit() else None
        except (ValueError, AttributeError):
            return None
    
    # 标准化格式
    formatted_data = [{
        "text": item.get("text", ""),
        "start_time": item.get("start", 0.0),
        "end_time": item.get("end", 0.0),
        "speaker_id": speaker_str_to_int(item.get("speaker_id", "SPEAKER_00"))
    } for item in transcript_data]
    
    return "".join([i["text"] for i in formatted_data]), formatted_data

# --- 主接口 ---

@router.post("/process", response_model=MeetingResponse)
async def process_meeting_audio(
    files: Optional[List[UploadFile]] = File(None),
    file_paths: Optional[str] = Form(None),
    audio_urls: Optional[str] = Form(None),
    audio_id: Optional[int] = Form(None),
    document_file: Optional[UploadFile] = File(None),
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
    temp_to_clean = []
    raw_text, transcript_data = "", []

    try:
        # 1. 输入源解析与预处理
        # 优先处理纯文本/文档
        if text_content:
            raw_text = text_content
        elif document_file:
            path = settings.TEMP_DIR / f"doc_{uuid.uuid4().hex}_{document_file.filename}"
            with open(path, "wb") as b: shutil.copyfileobj(document_file.file, b)
            temp_to_clean.append(str(path))
            raw_text = document_service.extract_text_from_file(str(path))
        
        # 处理音频输入
        else:
            audio_path, is_url = "", False
            # 多文件上传/路径解析
            if files:
                for idx, f in enumerate(files):
                    p = settings.TEMP_DIR / f"multi_{uuid.uuid4().hex}_{idx}_{f.filename}"
                    with open(p, "wb") as b: shutil.copyfileobj(f.file, b)
                    temp_to_clean.append(str(p))
                audio_path = temp_to_clean[0] # 这里简化逻辑：多文件并行目前仅演示首文件，如需全合并需ffmpeg
            elif audio_urls:
                audio_path = audio_urls.split(',')[0].strip().strip('"')
                is_url = audio_path.startswith("http")
            elif file_paths:
                audio_path = file_paths.split(',')[0].strip()

            # 2. 核心执行逻辑：并行流或传统流
            funasr_url = os.getenv("FUNASR_SERVICE_URL")
            pyannote_url = os.getenv("PYANNOTE_SERVICE_URL")

            if funasr_url and pyannote_url and asr_model == "funasr":
                logger.info("🚀 启动并行处理引擎...")
                raw_text, transcript_data = await handle_audio_parallel(audio_path, is_url, asr_model)
            
            # 降级/传统流程
            if not raw_text:
                asr_service = get_asr_service_by_name(asr_model)
                asr_res = asr_service.transcribe(audio_path)
                raw_text, transcript_data = asr_res.get("text", ""), asr_res.get("transcript", [])

            # 3. 声纹识别身份 (Voice Match)
            try:
                from app.services.voice_service import voice_service
                if voice_service.enabled and transcript_data and not is_url:
                    segments = voice_service.extract_speaker_segments(audio_path, transcript_data)
                    matched = voice_service.match_speakers(segments)
                    transcript_data = voice_service.replace_speaker_ids(transcript_data, matched)
            except Exception as ve:
                logger.warning(f"声纹匹配跳过: {ve}")

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

        # 渲染 Prompt 并调用 LLM
        llm_service = get_llm_service_by_name(llm_model)
        llm_service.temperature, llm_service.max_tokens = llm_temperature, llm_max_tokens
        
        template_config = prompt_template_service.get_template_config(template_id=template)
        final_prompt = prompt_template_service.render_prompt(template_config, raw_text, history_context, user_requirement)
        
        structured_data = llm_service.chat(final_prompt) if hasattr(llm_service, 'chat') else llm_service.generate_markdown(raw_text, "", template, user_requirement)
        
        # 格式化输出
        clean_md = structured_data.replace("```markdown", "").replace("```", "").strip()
        final_html = markdown.markdown(clean_md, extensions=['nl2br', 'tables'])

        return MeetingResponse(
            status="success",
            message="处理成功",
            raw_text=raw_text[:500],
            transcript=[TranscriptItem(**item) for item in transcript_data],
            html_content=final_html
        )

    except Exception as e:
        logger.error(f"❌ 处理异常: {e}\n{traceback.format_exc()}")
        return MeetingResponse(status="error", message=str(e), transcript=[])
    finally:
        cleanup_files(temp_to_clean)

# --- 其他接口 (Archive, Register, Hotwords) 逻辑已较精简，保持原有结构 ---

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
    file: UploadFile = File(..., description="员工录音文件(wav/mp3)"),
    name: str = Form(..., description="员工姓名"),
    employee_id: str = Form(..., description="员工工号(唯一标识)")
):
    """
    【声纹注册接口】供后端调用
    1. 接收音频流
    2. 转向量
    3. 存入 Chroma
    """
    temp_file_path = None
    
    try:
        # 1. 保存接收到的文件到临时目录
        # 即使后端传的是流，我们也得先存成文件给模型读
        file_ext = os.path.splitext(file.filename)[1] or ".wav"
        temp_filename = f"reg_{employee_id}_{uuid.uuid4()}{file_ext}"
        temp_file_path = settings.TEMP_DIR / temp_filename
        
        # 确保目录存在
        os.makedirs(settings.TEMP_DIR, exist_ok=True)
        
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"📥 收到注册请求: {name} (工号:{employee_id})")

        # 2. 延迟导入 voice_service（避免启动时加载）
        try:
            from app.services.voice_service import voice_service
        except ImportError as e:
            logger.error(f"❌ 声纹服务未安装或依赖缺失: {e}")
            return {
                "code": 500,
                "message": "声纹服务未安装，请联系管理员",
                "data": None
            }
        
        # 3. 调用服务提取向量
        vector = voice_service.extract_vector(str(temp_file_path))
        
        if not vector:
            return {
                "code": 400,
                "message": "音频质量过差或过短，无法提取声纹特征，请重录",
                "data": None
            }

        # 4. 存入库
        voice_service.save_identity(employee_id, name, vector)

        return {
            "code": 200,
            "message": "注册成功",
            "data": {
                "employee_id": employee_id,
                "name": name,
                "vector_dim": len(vector) # 返回维度供调试 (通常192)
            }
        }

    except Exception as e:
        logger.error(f"注册接口异常: {e}")
        return {"code": 500, "message": f"服务端内部错误: {str(e)}"}
        
    finally:
        # 4. 清理临时文件 (非常重要，否则硬盘会爆)
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except:
                pass


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