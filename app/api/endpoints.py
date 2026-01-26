import shutil
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from app.core.config import settings
from app.core.logger import logger
from app.schemas.task import MeetingResponse, ArchiveRequest, ArchiveResponse, TranscriptItem
from app.services.vector import vector_service
from app.services.asr_factory import get_asr_service
from app.services.llm_factory import get_llm_service
import markdown
from app.services.document import document_service 
# 延迟导入 voice_service，避免阻塞主服务启动
# from app.services.voice_service import voice_service
import uuid

# 创建路由器
router = APIRouter()

@router.post("/process", response_model=MeetingResponse)
async def process_meeting_audio(
    # ========== 输入源参数 ==========
    # 1. 音频文件流上传（单个或多个）
    file: Optional[UploadFile] = File(None),  # 单个文件（向后兼容）
    files: Optional[List[UploadFile]] = File(None),  # 多个文件（新增）
    
    # 2. 本地文件路径（用于测试或内部调用）
    file_path: Optional[str] = Form(None),  # 单个路径
    file_paths: Optional[str] = Form(None, description="多个本地文件路径（逗号分隔）"),  # 多个路径（新增）
    
    
    # 3. 音频URL（腾讯云ASR要求必须是可公网访问的URL）
    audio_url: Optional[str] = Form(None),
    
    # 4. 音频ID（从数据库获取）
    audio_id: Optional[int] = Form(None),
    
    # 5. 文档文件上传（Word/PDF）
    document_file: Optional[UploadFile] = File(None),

    # 6. 纯文本内容
    text_content: Optional[str] = Form(None),

    # ========== 模板参数 ==========
    # 模板 ID (或者是本地文件的绝对路径)
    template_id: str = Form("default"),
    
    # 动态模板内容（JSON字符串，优先级高于template_id）
    prompt_template: Optional[str] = Form(None, description="自定义提示词模板（JSON格式）"),

    # ========== 用户需求参数 ==========
    # 自定义指令（用户对纪要生成的特殊要求）
    custom_instruction: Optional[str] = Form(None, description="用户对纪要生成的特殊要求"),
    
    # 用户需求（新增，更明确的命名）
    user_requirement: Optional[str] = Form(None, description="用户的具体需求"),
    
    # ========== 历史会议参数（新增）==========
    history_meeting_ids: Optional[str] = Form(None, description="历史会议ID列表（逗号分隔）"),
    history_mode: str = Form("auto", description="历史会议处理模式（auto/retrieval/summary）"),
    
    # ========== 模型选择参数（新增）==========
    llm_model: str = Form("auto", description="LLM模型（auto/deepseek/qwen3）"),
    llm_temperature: float = Form(0.7, description="生成温度（0.0-1.0）"),
    llm_max_tokens: int = Form(2000, description="最大生成长度"),
    
    # ASR模型选择（新增）
    asr_model: str = Form("auto", description="ASR模型（auto/tencent/funasr）"),
):
    """
    全能接口: 支持 音频 / 文档 / 纯文本 三大类输入
    
    ✨ 新功能：支持多音频合并处理
    - 单个文件：file 或 file_path
    - 多个文件：files 或 file_paths（逗号分隔）
    """
    temp_file_path = None  # 需要清理的临时文件路径
    temp_files = []  # 多音频临时文件列表
    raw_text = ""
    transcript_data = []  # 逐字稿数据

    try:
        # ========== 情况 A: 处理音频 ==========
        # 检测是否为多音频模式
        is_multi_audio = False
        audio_paths = []
        
        # 判断1: 多个文件上传
        if files and len(files) > 0:
            is_multi_audio = True
            for idx, upload_file in enumerate(files):
                if upload_file.filename:
                    temp_path = settings.TEMP_DIR / f"multi_{idx}_{upload_file.filename}"
                    with open(temp_path, "wb") as buffer:
                        shutil.copyfileobj(upload_file.file, buffer)
                    audio_paths.append(str(temp_path))
                    temp_files.append(temp_path)
                    logger.info(f"💾 音频 [{idx+1}/{len(files)}] 已保存: {temp_path}")
        
        # 判断2: 多个文件路径（逗号分隔）
        elif file_paths:
            is_multi_audio = True
            paths = [p.strip() for p in file_paths.split(',') if p.strip()]
            for path in paths:
                if not os.path.exists(path):
                    return MeetingResponse(
                        status="failed",
                        message=f"文件不存在: {path}",
                        transcript=[]
                    )
                audio_paths.append(path)
            logger.info(f"📂 使用多个本地文件: 共 {len(audio_paths)} 个")
        
        # === 多音频处理分支 ===
        if is_multi_audio and audio_paths:
            logger.info(f"🎵 多音频模式: 共 {len(audio_paths)} 个音频文件")
            
            # 获取ASR服务
            asr_service = get_asr_service(asr_model)
            logger.info(f"🎤 使用ASR模型: {asr_model}")
            
            # 逐个识别并合并
            current_speaker_offset = 0
            
            for idx, audio_path in enumerate(audio_paths):
                logger.info(f"🎤 [{idx+1}/{len(audio_paths)}] 识别中: {os.path.basename(audio_path)}")
                
                asr_result = asr_service.transcribe(audio_path)
                
                if not asr_result or not asr_result.get("text"):
                    logger.warning(f"⚠️ 音频 [{idx+1}] 识别结果为空，跳过")
                    continue
                
                # 重新编号 speaker_id
                transcript = asr_result.get("transcript", [])
                if transcript:
                    max_speaker_id = 0
                    for item in transcript:
                        if item.get("speaker_id"):
                            original_id = item["speaker_id"]
                            item["speaker_id"] = original_id + current_speaker_offset
                            max_speaker_id = max(max_speaker_id, item["speaker_id"])
                    
                    if max_speaker_id > 0:
                        current_speaker_offset = max_speaker_id
                    
                    transcript_data.extend(transcript)
                    logger.info(f"✅ 音频 [{idx+1}] 识别成功: {len(transcript)} 条")
            
            if not transcript_data:
                return MeetingResponse(
                    status="failed",
                    message="所有音频识别结果均为空",
                    transcript=[]
                )
            
            # 合并所有文本
            raw_text = "\n".join([item.get("text", "") for item in transcript_data])
            logger.info(f"📝 多音频合并完成: {len(audio_paths)} 个文件, 总长度 {len(raw_text)} 字")
        
        # === 单音频处理分支（原有逻辑） ===
        elif file or file_path or audio_id or audio_url:
            # ✅ 使用print确保终端显示
            print(f"\n{'='*80}")
            print(f"📨 收到新的音频处理请求")
            print(f"{'='*80}")
            import sys
            sys.stderr.flush()
            sys.stdout.flush()
            
            logger.info(f"📨 收到音频处理请求: 模板={template_id}")
            
            target_audio_path = ""

            # 分支 1: 传了文件流 - 直接保存
            if file:
                temp_file_path = settings.TEMP_DIR / f"upload_{file.filename}"
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                target_audio_path = str(temp_file_path)
                logger.info(f"💾 音频流已保存: {target_audio_path}")
            
            # 分支 2: 传了本地文件路径 - 直接使用（用于测试或内部调用）
            elif file_path:
                import os
                if not os.path.exists(file_path):
                    return MeetingResponse(
                        status="failed",
                        message=f"文件不存在: {file_path}",
                        transcript=[]
                    )
                target_audio_path = file_path
                logger.info(f"📂 使用本地文件路径: {target_audio_path}")
            
            # 分支 3: 传了音频ID - 从数据库获取并下载
            elif audio_id:
                from app.services.download import audio_download_service
                target_audio_path = audio_download_service.get_file_path_from_db(audio_id)
                
                if not target_audio_path:
                    return MeetingResponse(
                        status="failed",
                        message=f"无法从数据库获取或下载音频: ID={audio_id}",
                        transcript=[]
                    )
                # 标记为临时文件，需要清理
                temp_file_path = target_audio_path
                logger.info(f"📥 从数据库获取音频并下载: ID={audio_id}, 路径={target_audio_path}")
            
            # 分支 4: 传了音频URL - 直接使用（腾讯云ASR要求）
            # 也支持音频地址 (支持 URL 或 本地路径)
            elif audio_url:
                # 1. 清洗输入 (去掉可能存在的引号和空格，防止 copy 路径带引号)
                clean_path = audio_url.strip().strip('"').strip("'").strip()
                
                is_url = clean_path.startswith(("http://", "https://"))
                is_local_file = os.path.exists(clean_path)
                
                # 2. 根据当前的 ASR 服务类型做校验
                if settings.ASR_SERVICE_TYPE == 'tencent':
                    # 【腾讯云模式】必须是 URL
                    if not is_url:
                        return MeetingResponse(
                            status="failed",
                            message=f"模式错误: 当前使用【腾讯云】，必须提供公网 URL，不支持本地路径: {clean_path}",
                            transcript=[]
                        )
                    target_audio_path = clean_path
                    logger.info(f"🔗 [腾讯云] 使用音频URL: {target_audio_path}")

                else:
                    # 【本地 FunASR 模式】支持 URL + 本地文件
                    if is_url:
                        target_audio_path = clean_path
                        logger.info(f"🔗 [本地模式] 识别为网络地址: {target_audio_path}") # Service层会自动下载
                    
                    elif is_local_file:
                        if os.path.isdir(clean_path):
                            return MeetingResponse(status="failed", message="路径是一个文件夹，请指定具体文件", transcript=[])
                        
                        target_audio_path = clean_path
                        logger.info(f"📂 [本地模式] 识别为本地文件: {target_audio_path}")
                    
                    else:
                        # 既不是 URL，本地也没这个文件
                        return MeetingResponse(
                            status="failed",
                            message=f"无效路径: 系统找不到文件 '{clean_path}'，且不是 http 链接。",
                            transcript=[]
                        )
            
            # 如果是本地文件，验证文件大小
            if not target_audio_path.startswith(("http://", "https://")):
                if not os.path.exists(target_audio_path):
                    return MeetingResponse(
                        status="failed",
                        message=f"音频文件不存在: {target_audio_path}",
                        transcript=[]
                    )
                
                file_size_mb = os.path.getsize(target_audio_path) / (1024 * 1024)
                if file_size_mb > settings.MAX_FILE_SIZE_MB:
                    return MeetingResponse(
                        status="failed",
                        message=f"音频文件过大: {file_size_mb:.2f}MB，最大允许: {settings.MAX_FILE_SIZE_MB}MB",
                        transcript=[]
                    )
                logger.info(f"📊 音频文件大小: {file_size_mb:.2f}MB")

            # 获取 ASR 服务（动态选择）⭐
            try:
                from app.services.asr_factory import get_asr_service_by_name
                asr_service = get_asr_service_by_name(asr_model)
                logger.info(f"🎤 使用ASR模型: {asr_model}")
            except Exception as e:
                return MeetingResponse(
                    status="failed", 
                    message=f"ASR服务初始化失败: {str(e)}",
                    transcript=[]
                )
            
            # 调用 ASR 服务听写
            asr_result = asr_service.transcribe(target_audio_path)
            raw_text = asr_result.get("text", "")
            transcript_data = asr_result.get("transcript", [])
            
            if not raw_text:
                return MeetingResponse(
                    status="failed", 
                    message="语音识别结果为空",
                    transcript=[]
                )

        # --- 情况 B: 处理文档（Word/PDF）---
        elif document_file:
            logger.info(f"📄 收到文档处理请求: 文件名={document_file.filename}, 模板={template_id}")
            
            file_ext = os.path.splitext(document_file.filename)[1].lower()
            if file_ext not in ['.docx', '.pdf', '.txt']:
                return MeetingResponse(
                    status="failed",
                    message=f"不支持的文档格式: {file_ext}，仅支持 .docx, .pdf, .txt",
                    transcript=[]
                )
            
            temp_file_path = settings.TEMP_DIR / f"doc_{document_file.filename}"
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(document_file.file, buffer)
            logger.info(f"💾 文档已保存: {temp_file_path}")
            
            # 使用 document_service 提取文本
            raw_text = document_service.extract_text_from_file(str(temp_file_path))
            
            if not raw_text:
                return MeetingResponse(
                    status="failed",
                    message="文档解析失败或文档内容为空",
                    transcript=[]
                )
            logger.info(f"✅ 文档解析完成，文本长度: {len(raw_text)}")

        # --- 情况 C: 处理纯文本 ---
        elif text_content:
            logger.info(f"📨 收到纯文本请求: 长度={len(text_content)}")
            raw_text = text_content
            
        # --- 情况 D: 啥都没传 ---
        else:
            return MeetingResponse(
                status="failed", 
                message="请提供输入: 音频文件/URL/ID, 文档或文本内容",
                transcript=[]
            )

        # ---------------------------------------------------------
        # 2. 【核心修复】解析模板内容 (如果是本地文件路径)
        # ---------------------------------------------------------
        real_template_content = template_id  # 默认值
        
        # 🧹 第一步：清洗路径 (去除可能存在的引号和空格)
        if template_id:
            clean_path = template_id.strip().strip('"').strip("'")
        else:
            clean_path = ""

        # 🖨️ 强制打印调试信息 (请关注控制台输出)
        if clean_path and ".docx" in clean_path:
            logger.info(f"🔍 [调试] 正在检查路径: [{clean_path}]")
            logger.info(f"🔍 [调试] 文件是否存在: {os.path.exists(clean_path)}")

        # 📂 第二步：判断并读取
        if clean_path and len(clean_path) > 3 and clean_path.lower().endswith(('.docx', '.pdf', '.txt')):
            if os.path.exists(clean_path):
                logger.info(f"📂 检测到本地模板文件: {clean_path}，正在读取...")
                
                # 调用 DocumentService 读取模板文件
                extracted_template = document_service.extract_text_from_file(clean_path)
                
                if extracted_template:
                    # ★★★ 关键点：这里把路径换成了真实内容 ★★★
                    real_template_content = extracted_template
                    logger.info(f"✅ 成功读取本地模板内容，字数: {len(real_template_content)}")
                    # 打印前50个字看看是不是真的读到了
                    logger.info(f"📝 模板预览: {real_template_content[:50]}...")
                else:
                    logger.warning(f"⚠️ 模板文件读取为空")
            else:
                logger.warning(f"⚠️ 路径看起来像文件，但系统找不到: {clean_path}")

        # ---------------------------------------------------------
        # 历史会议处理部分（新增）⭐
        # ---------------------------------------------------------
        history_context = None
        
        # 合并用户需求（custom_instruction 和 user_requirement）
        final_user_requirement = user_requirement or custom_instruction
        
        if history_meeting_ids:
            # 解析历史会议ID列表
            meeting_ids = [
                mid.strip() 
                for mid in history_meeting_ids.split(",") 
                if mid.strip()
            ]
            
            if meeting_ids:
                from app.services.meeting_history import meeting_history_service
                
                # 判断使用哪种模式
                mode = meeting_history_service.determine_mode(
                    meeting_ids=meeting_ids,
                    user_requirement=final_user_requirement,
                    history_mode=history_mode
                )
                
                logger.info(f"📚 处理历史会议: {len(meeting_ids)} 个, 模式: {mode}")
                
                try:
                    if mode == "retrieval":
                        # 检索模式：精确查询
                        history_context = await meeting_history_service.process_by_retrieval(
                            meeting_ids=meeting_ids,
                            user_requirement=final_user_requirement,
                            current_transcript=raw_text,
                            llm_model=llm_model
                        )
                    else:
                        # 总结模式：分块汇总
                        history_context = await meeting_history_service.process_by_summary(
                            meeting_ids=meeting_ids,
                            user_requirement=final_user_requirement,
                            llm_model=llm_model
                        )
                    
                    logger.info(f"✅ 历史会议处理完成: {mode} 模式")
                    
                except Exception as e:
                    logger.error(f"❌ 历史会议处理失败: {e}")
                    # 不影响主流程，继续处理
                    history_context = None
        
        # ---------------------------------------------------------
        # LLM 处理部分
        # ---------------------------------------------------------

        try:
            # 动态选择模型（新增）⭐
            from app.services.llm_factory import get_llm_service_by_name
            llm_service = get_llm_service_by_name(llm_model)
            
            # 设置 LLM 参数
            if hasattr(llm_service, 'temperature'):
                llm_service.temperature = llm_temperature
            if hasattr(llm_service, 'max_tokens'):
                llm_service.max_tokens = llm_max_tokens
            
        except Exception as e:
            logger.error(f"❌ LLM服务初始化失败: {e}")
            # ... (错误处理保持不变)
            transcript_items = []
            if transcript_data:
                from app.schemas.task import TranscriptItem
                transcript_items = [
                    TranscriptItem(**item) for item in transcript_data
                ]
            return MeetingResponse(
                status="failed",
                message=f"LLM服务初始化失败: {str(e)}",
                raw_text=raw_text[:500],
                transcript=transcript_items
            )

        # 1. 使用动态模板渲染（新增）⭐
        from app.services.prompt_template import prompt_template_service
        
        # 获取模板配置（优先使用自定义模板）
        template_config = prompt_template_service.get_template_config(
            prompt_template=prompt_template,
            template_id=template_id if not real_template_content or real_template_content == template_id else "default"
        )
        
        # 渲染最终的提示词
        final_prompt = prompt_template_service.render_prompt(
            template_config=template_config,
            current_transcript=raw_text,
            history_context=history_context,
            user_requirement=final_user_requirement
        )
        
        logger.info(f"📝 提示词渲染完成，长度: {len(final_prompt)}")
        
        # 2. 调用 LLM 生成
        # 注意：这里直接调用 chat 方法，而不是 generate_markdown
        # 因为提示词已经包含了所有上下文
        try:
            if hasattr(llm_service, 'chat'):
                structured_data = llm_service.chat(final_prompt)
            else:
                # 降级：使用原有的 generate_markdown 方法
                logger.warning("⚠️ LLM 服务没有 chat 方法，使用原有逻辑")
                
                # RAG 分析（原有逻辑）
                rag_analysis = llm_service.judge_rag(raw_text, template_id)
                need_rag = rag_analysis.get("need_rag", False)
                search_query = rag_analysis.get("search_query", "")

                # 向量检索
                context_info = "" 
                if need_rag and search_query:
                    if vector_service and vector_service.is_available():
                        context_info = vector_service.search_similar(search_query)
                        logger.info(f"📚 基于 '{search_query}' 检索到历史上下文")
                    else:
                        logger.warning("⚠️ 向量服务不可用，跳过历史检索")

                # 生成
                structured_data = llm_service.generate_markdown(
                    raw_text=raw_text, 
                    context=context_info,
                    template_id=real_template_content,
                    custom_instruction=final_user_requirement
                )
        except Exception as e:
            logger.error(f"❌ LLM 生成失败: {e}")
            raise

        final_html = ""
        if structured_data:
            try:
                # extensions=['nl2br'] 确保换行符会被转为 <br>
                clean_md = structured_data.replace("```markdown", "").replace("```", "").strip()
                final_html = markdown.markdown(clean_md, extensions=['nl2br', 'tables'])
            except Exception as e:
                logger.error(f"HTML转换失败: {e}")
                final_html = f"<p>{structured_data}</p>" # 降级处理
        
        # 构建返回
        transcript_items = []
        if transcript_data:
            from app.schemas.task import TranscriptItem
            transcript_items = [
                TranscriptItem(
                    text=item.get("text", ""),
                    start_time=item.get("start_time", 0.0),
                    end_time=item.get("end_time", 0.0),
                    speaker_id=item.get("speaker_id")
                )
                for item in transcript_data
            ]

        logger.info("✅ 任务完成")

        return MeetingResponse(
            status="success",
            message="处理成功",
            raw_text=raw_text[:500],
            transcript=transcript_items,
            need_rag=False,  # 新逻辑下不需要这个字段
            html_content=final_html,
            usage_tokens=0
        )

    except Exception as e:
        logger.error(f"❌ 接口处理异常: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return MeetingResponse(
            status="error", 
            message=f"服务端错误: {str(e)}",
            transcript=[]
        )
    
    finally:
        # 清理临时文件
        # 1. 单文件清理
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"🧹 临时文件已清理: {temp_file_path}")
            except Exception as e:
                logger.warning(f"⚠️ 清理临时文件失败: {e}")
        
        # 2. 多文件清理
        for temp_path in temp_files:
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                    logger.info(f"🧹 临时文件已清理: {temp_path}")
                except Exception as e:
                    logger.warning(f"⚠️ 清理临时文件失败: {temp_path}, {e}")

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