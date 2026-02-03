import shutil
import os
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from app.core.config import settings
from app.core.logger import logger
from app.schemas.task import MeetingResponse, ArchiveRequest, ArchiveResponse, TranscriptItem
from app.services.vector import vector_service
from app.services.asr_factory import get_asr_service_by_name
from app.services.llm_factory import get_llm_service, get_llm_service_by_name
import markdown
from app.services.document import document_service 
# 延迟导入 voice_service，避免阻塞主服务启动
# from app.services.voice_service import voice_service
import uuid

# 创建路由器
router = APIRouter()

@router.post("/process", response_model=MeetingResponse)
async def process_meeting_audio(
    # ========== 输入源参数（以下7种方式任选其一）==========
    
    files: Optional[List[UploadFile]] = File(
        None, 
        description="🎵 音频文件上传：\n• 支持格式：mp3/wav/m4a/mp4等\n• 支持单个或多个文件\n• 多个文件会自动合并处理"
    ),
    
    file_paths: Optional[str] = Form(
        None, 
        description="📂 本地文件路径：\n• 单个：test_audio/meeting.mp3\n• 多个：audio1.mp3,audio2.mp3（逗号分隔）"
    ),
    
    audio_urls: Optional[str] = Form(
        None, 
        description="🌐 音频URL地址：\n• 要求：可公网访问的URL（腾讯云ASR需要）\n• 单个：http://example.com/audio.mp3\n• 多个：url1,url2（逗号分隔）"
    ),
    
    audio_id: Optional[int] = Form(
        None, 
        description="🔢 数据库音频ID：用于处理已存储到数据库的历史音频"
    ),
    
    document_file: Optional[UploadFile] = File(
        None, 
        description="📄 文档文件上传：\n• 支持格式：Word(.docx) / PDF(.pdf) / 文本(.txt)\n• 直接提取文字生成纪要（不需要语音识别）"
    ),

    text_content: Optional[str] = Form(
        None, 
        description="📝 纯文本内容：\n• 直接输入会议文本或已转录好的内容\n• 跳过语音识别步骤，直接生成纪要"
    ),

    # ========== 模板参数 ==========
    template: str = Form(
        "default", 
        description="📋 模板配置：\n• 预设模板ID：default（标准）/ simple（简洁）/ detailed（详细）\n• 文档路径：D:\\模板.docx（自定义格式）\n• JSON字符串：自定义提示词\n• 纯文本：直接的提示词内容"
    ),

    # ========== 用户需求参数 ==========
    user_requirement: Optional[str] = Form(
        None, 
        description="✨ 特殊要求（可选）：对生成纪要的个性化需求，如\"重点关注预算讨论\"、\"简化技术细节\"等"
    ),
    
    # ========== 历史会议参数 ==========
    history_meeting_ids: Optional[str] = Form(
        None, 
        description="🔗 关联历史会议（可选）：\n• 格式：会议ID列表，逗号分隔\n• 示例：100,101,102\n• 用途：生成纪要时参考历史会议内容"
    ),
    
    history_mode: str = Form(
        "auto", 
        description="📚 历史处理模式：\n• auto：自动判断（推荐）\n• retrieval：检索模式（查找相关历史内容）\n• summary：总结模式（提供历史会议总结）"
    ),
    
    # ========== 模型配置参数 ==========
    llm_model: str = Form(
        "auto", 
        description="🤖 LLM模型选择：\n• auto：自动选择（使用配置文件设置）\n• deepseek：DeepSeek API\n• qwen3：本地Qwen3模型"
    ),
    
    llm_temperature: float = Form(
        0.7, 
        description="🌡️ 生成温度（0.0-1.0）：\n• 0.3：更保守，输出更确定\n• 0.7：平衡（推荐）\n• 1.0：更有创造性，输出更多样"
    ),
    
    llm_max_tokens: int = Form(
        2000, 
        description="📏 最大生成长度：生成纪要的最大字数（token数）"
    ),
    
    asr_model: str = Form(
        "auto", 
        description="🎤 语音识别模型：\n• auto：自动选择（使用配置文件设置）\n• funasr：本地FunASR（推荐）\n• tencent：腾讯云ASR"
    ),
):
    """
    ## 🎯 会议纪要生成接口
    
    **功能：** 将音频/文档/文本转换为结构化的会议纪要
    
    ---
    
    ### 📥 输入方式（以下7种任选其一）
    
    | 方式 | 参数 | 说明 | 场景 |
    |-----|------|------|------|
    | 🎵 上传音频 | `files` | 支持mp3/wav/m4a等，可多个 | 常用：会议录音 |
    | 📂 本地路径 | `file_paths` | 逗号分隔多个路径 | 开发测试 |
    | 🌐 音频URL | `audio_urls` | 公网可访问URL | 腾讯云ASR |
    | 🔢 数据库ID | `audio_id` | 已存储的音频ID | 历史音频 |
    | 📄 上传文档 | `document_file` | Word/PDF/TXT | 已有文字记录 |
    | 📝 纯文本 | `text_content` | 直接输入文本 | 已转录内容 |
    
    ---
    
    ### 🎨 输出格式
    
    **模板参数** `template`：
    - 预设模板：`default`（标准）/ `simple`（简洁）/ `detailed`（详细）
    - 自定义文档：上传 `.docx` / `.pdf` 模板文件路径
    - 自定义提示词：直接写提示词内容
    
    ---
    
    ### ⚙️ 可选配置
    
    - `user_requirement`：特殊要求（如"重点关注预算"）
    - `history_meeting_ids`：关联历史会议ID
    - `history_mode`：历史处理模式（auto/retrieval/summary）
    - `llm_model`：选择LLM模型（auto/deepseek/qwen3）
    - `asr_model`：选择ASR模型（auto/funasr/tencent）
    
    ---
    
    ### 💡 使用示例
    
    **示例1：上传单个音频**
    ```python
    files = [meeting.mp3]
    template = "default"
    ```
    
    **示例2：上传多个音频（自动合并）**
    ```python
    files = [part1.mp3, part2.mp3, part3.mp3]
    template = "default"
    ```
    
    **示例3：自定义模板和需求**
    ```python
    files = [meeting.mp3]
    template = "D:\\模板\\周例会模板.docx"
    user_requirement = "重点关注预算讨论和人员调整"
    ```
    
    **示例4：关联历史会议**
    ```python
    files = [meeting.mp3]
    template = "default"
    history_meeting_ids = "100,101,102"
    history_mode = "retrieval"
    ```
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
                    # 使用UUID前缀避免并发冲突
                    temp_path = settings.TEMP_DIR / f"multi_{uuid.uuid4().hex}_{idx}_{upload_file.filename}"
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
            asr_service = get_asr_service_by_name(asr_model)
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
                        if item.get("speaker_id") is not None:
                            original_id = item["speaker_id"]
                            # 统一转换为整数处理
                            if isinstance(original_id, str):
                                # 如果是字符串（如 "spk0"），提取数字部分
                                try:
                                    original_id = int(''.join(filter(str.isdigit, original_id)) or "0")
                                except:
                                    original_id = 0
                            else:
                                original_id = int(original_id)
                            
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
            
            # ---------------------------------------------
            # 可选：调用独立 Pyannote 服务进行说话人分离（方案B）
            # 仅在配置了 PYANNOTE_SERVICE_URL 且只有一个音频文件时启用
            # （多音频文件时，Pyannote 需要分别处理每个文件，这里简化处理）
            # ---------------------------------------------
            if len(audio_paths) == 1:
                try:
                    from app.services.pyannote_service import get_pyannote_service
                    pyannote_service = get_pyannote_service()
                    
                    if pyannote_service.is_available() and transcript_data:
                        single_audio_path = audio_paths[0]
                        if not single_audio_path.startswith(("http://", "https://")):
                            logger.info("🎤 使用独立 Pyannote 服务优化说话人分离（方案B）")
                            transcript_data = pyannote_service.diarize(
                                audio_path=single_audio_path,
                                transcript=transcript_data,
                            )
                        else:
                            logger.info("ℹ️ 目标音频为 URL，当前 Pyannote 仅支持本地文件，跳过")
                    elif not pyannote_service.is_available():
                        logger.info("ℹ️ 未配置 PYANNOTE_SERVICE_URL，跳过 Pyannote 分离")
                    elif not transcript_data:
                        logger.info("ℹ️ transcript 为空，跳过 Pyannote 分离")
                except Exception as e:
                    logger.warning(f"⚠️ 调用 Pyannote 服务失败，保持原有说话人结果: {e}")
            else:
                logger.info(f"ℹ️ 多音频模式（{len(audio_paths)} 个文件），当前版本暂不支持 Pyannote 优化")
        
        # === 单音频处理分支（原有逻辑） ===
        # 处理单个文件/URL/ID的情况
        elif (files and len(files) == 1) or file_paths or audio_id or audio_urls:
            # ✅ 使用print确保终端显示
            print(f"\n{'='*80}")
            print(f"📨 收到新的音频处理请求")
            print(f"{'='*80}")
            import sys
            sys.stderr.flush()
            sys.stdout.flush()
            
            logger.info(f"📨 收到音频处理请求: 模板={template}")
            
            target_audio_path = ""

            # 分支 1: 传了文件流 - 直接保存
            if files and len(files) == 1:
                upload_file = files[0]
                # 使用UUID前缀避免并发冲突
                temp_file_path = settings.TEMP_DIR / f"upload_{uuid.uuid4().hex}_{upload_file.filename}"
                with open(temp_file_path, "wb") as buffer:
                    shutil.copyfileobj(upload_file.file, buffer)
                target_audio_path = str(temp_file_path)
                logger.info(f"💾 音频流已保存: {target_audio_path}")
            
            # 分支 2: 传了本地文件路径 - 直接使用（用于测试或内部调用）
            elif file_paths:
                # 支持单个或多个路径（如果是多个，只取第一个）
                paths = [p.strip() for p in file_paths.split(',') if p.strip()]
                target_path = paths[0] if paths else None
                
                if not target_path:
                    return MeetingResponse(
                        status="failed",
                        message="file_paths 参数为空",
                        transcript=[]
                    )
                
                file_path = target_path  # 临时变量，用于后续处理
                
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
            elif audio_urls:
                # 支持单个或多个URL（如果是多个，只取第一个）
                urls = [url.strip() for url in audio_urls.split(',') if url.strip()]
                audio_url = urls[0] if urls else None
                
                if not audio_url:
                    return MeetingResponse(
                        status="failed",
                        message="audio_urls 参数为空",
                        transcript=[]
                    )
                
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

            # ---------------------------------------------
            # 可选：调用独立 Pyannote 服务进行说话人分离（方案B）
            # 仅在配置了 PYANNOTE_SERVICE_URL 时启用
            # ---------------------------------------------
            try:
                from app.services.pyannote_service import get_pyannote_service
                pyannote_service = get_pyannote_service()

                if pyannote_service.is_available() and transcript_data and not target_audio_path.startswith(("http://", "https://")):
                    logger.info("🎤 使用独立 Pyannote 服务优化说话人分离（方案B）")
                    transcript_data = pyannote_service.diarize(
                        audio_path=target_audio_path,
                        transcript=transcript_data,
                    )
                else:
                    if not pyannote_service.is_available():
                        logger.info("ℹ️ 未配置 PYANNOTE_SERVICE_URL，跳过 Pyannote 分离")
                    elif not transcript_data:
                        logger.info("ℹ️ transcript 为空，跳过 Pyannote 分离")
                    else:
                        logger.info("ℹ️ 目标音频为 URL，当前 Pyannote 仅支持本地文件，跳过")
            except Exception as e:
                logger.warning(f"⚠️ 调用 Pyannote 服务失败，保持原有说话人结果: {e}")
            
            if not raw_text:
                return MeetingResponse(
                    status="failed", 
                    message="语音识别结果为空",
                    transcript=[]
                )

        # --- 情况 B: 处理文档（Word/PDF）---
        elif document_file:
            logger.info(f"📄 收到文档处理请求: 文件名={document_file.filename}, 模板={template}")
            
            file_ext = os.path.splitext(document_file.filename)[1].lower()
            if file_ext not in ['.docx', '.pdf', '.txt']:
                return MeetingResponse(
                    status="failed",
                    message=f"不支持的文档格式: {file_ext}，仅支持 .docx, .pdf, .txt",
                    transcript=[]
                )
            
            # 使用UUID前缀避免并发冲突
            temp_file_path = settings.TEMP_DIR / f"doc_{uuid.uuid4().hex}_{document_file.filename}"
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
        # 2. 模板处理（已移到 prompt_template_service 中统一处理）
        # ---------------------------------------------------------
        # 现在 prompt_template_service.get_template_config 已经支持文档路径
        # 所以这里不需要额外处理了

        # ---------------------------------------------------------
        # 历史会议处理部分（新增）⭐
        # ---------------------------------------------------------
        history_context = None
        
        # 用户需求（已在向后兼容处理中合并）
        final_user_requirement = user_requirement
        
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
        
        # 获取模板配置（统一使用 template 参数）
        # template 可以是：模板ID、文档路径、JSON字符串或纯文本
        template_config = prompt_template_service.get_template_config(
            prompt_template=None,  # 不再使用废弃参数
            template_id=template    # 使用新的统一参数
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

                # 生成（使用模板配置中的模板内容或template_id）
                # 如果模板配置包含 prompt_template，使用它；否则使用 template_id
                template_to_use = template_config.get("prompt_template", template_id)
                
                structured_data = llm_service.generate_markdown(
                    raw_text=raw_text, 
                    context=context_info,
                    template_id=template_to_use,
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