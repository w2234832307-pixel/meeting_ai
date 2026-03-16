"""
动态提示词模板渲染服务
支持Jinja2模板语法和动态变量替换
"""
from typing import Dict, Any, Optional
from jinja2 import Template, TemplateError
import json
import os
from pathlib import Path
import uuid
from app.core.logger import logger
from app.prompts.templates import get_default_template


class PromptTemplateService:
    """提示词模板渲染服务"""
    
    @staticmethod
    def _load_mappings() -> Optional[str]:
        """
        从 hotwords.json 加载映射表并格式化为提示词
        
        Returns:
            格式化的映射指导文本，如果加载失败返回None
        """
        try:
            # 查找 hotwords.json 文件
            hotwords_paths = [
                Path("funasr_standalone/hotwords.json"),  # 相对路径
                Path(__file__).parent.parent.parent / "funasr_standalone" / "hotwords.json",  # 绝对路径
            ]
            
            hotwords_file = None
            for path in hotwords_paths:
                if path.exists():
                    hotwords_file = path
                    break
            
            if not hotwords_file:
                logger.debug("⚠️ 未找到 hotwords.json，跳过映射加载")
                return None
            
            # 读取并解析
            with open(hotwords_file, 'r', encoding='utf-8') as f:
                hotwords_config = json.load(f)
            
            mappings = hotwords_config.get("mappings", {})
            
            if not mappings:
                return None
            
            # 格式化映射表为提示词
            mapping_parts = ["=" * 60]
            mapping_parts.append("🚨🚨🚨 【名称标准化映射表 - 必须严格执行】 🚨🚨🚨")
            mapping_parts.append("=" * 60)
            mapping_parts.append("⚠️⚠️⚠️ 这是最高优先级要求！必须将以下所有口语化表达替换为标准名称！\n")
            
            for category, mapping_dict in mappings.items():
                if mapping_dict:
                    mapping_parts.append(f"【{category}映射规则 - 必须100%执行】")
                    for oral, standard in mapping_dict.items():
                        # 使用更醒目的格式
                        mapping_parts.append(f"  ❌ \"{oral}\" (禁止使用) ➜ ✅ \"{standard}\" (必须使用)")
                    mapping_parts.append("")
            
            mapping_parts.append("📋 执行规则（不可违反）：")
            mapping_parts.append("✓ 规则1：转录文本中的左侧口语化表达 ➜ 必须100%替换为右侧标准名称")
            mapping_parts.append("✓ 规则2：整篇纪要中不允许出现映射表左侧的任何口语化表达")
            mapping_parts.append("✓ 规则3：所有人名必须使用标准全名，不允许使用昵称、简称")
            mapping_parts.append("✓ 规则4：所有项目名必须使用标准全称，不允许使用口语化简称")
            mapping_parts.append("✓ 规则5：遇到映射表中没有的新称呼，也应该尝试推断其标准名称\n")
            mapping_parts.append("=" * 60)
            mapping_parts.append("🔥 请在生成每一句话时都检查是否应用了映射规则！")
            mapping_parts.append("=" * 60 + "\n")
            
            return "\n".join(mapping_parts)
            
        except Exception as e:
            logger.warning(f"⚠️ 加载映射表失败: {e}")
            return None
    
    @staticmethod
    def render_prompt(
        template_config: Dict[str, Any],
        current_transcript: str,
        history_context: Optional[Dict] = None,
        user_requirement: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        渲染提示词模板
        
        Args:
            template_config: 模板配置
            current_transcript: 当前会议转录文本
            history_context: 历史会议上下文
            user_requirement: 用户需求
            **kwargs: 其他动态变量
        
        Returns:
            渲染后的提示词
        """
        try:
            # 获取模板内容
            prompt_template = template_config.get("prompt_template", "")
            variables = template_config.get("variables", {})
            dynamic_sections = template_config.get("dynamic_sections", {})
            
            # === 构建动态部分 ===
            
            # 1. 历史会议部分
            history_section = ""
            if history_context:
                history_template_str = dynamic_sections.get("history_section", "")
                
                if history_template_str:
                    # 构建历史内容
                    history_content = PromptTemplateService._build_history_content(
                        history_context
                    )
                    
                    if history_content:
                        try:
                            history_template = Template(history_template_str)
                            history_section = history_template.render(
                                history_content=history_content
                            )
                        except TemplateError as e:
                            logger.error(f"❌ 历史部分模板渲染失败: {e}")
            
            # 2. 用户需求部分
            requirement_section = ""
            if user_requirement and user_requirement.strip():
                requirement_template_str = dynamic_sections.get(
                    "requirement_section", ""
                )
                
                if requirement_template_str:
                    try:
                        requirement_template = Template(requirement_template_str)
                        requirement_section = requirement_template.render(
                            user_requirement=user_requirement
                        )
                    except TemplateError as e:
                        logger.error(f"❌ 需求部分模板渲染失败: {e}")
            
            # 3. 映射表部分（名称标准化）
            mapping_section = ""
            mappings_text = PromptTemplateService._load_mappings()
            if mappings_text:
                mapping_section = mappings_text
                logger.info("✅ 已加载名称映射表到提示词")
            
            # 4. 参考文档部分（用户上传的附件）
            reference_document_section = ""
            reference_document = kwargs.get("reference_document")
            if reference_document and reference_document.strip():
                # 构建参考文档部分
                reference_document_section = f"""
## 📎 参考文档（用户上传的附件）
以下内容来自用户上传的参考文档，请结合这些信息生成会议纪要：

{reference_document[:5000]}  # 限制长度避免超出token限制

---
"""
                logger.info(f"✅ 已加载参考文档到提示词，长度: {len(reference_document)} 字符")
            
            # === 渲染最终 Prompt ===
            try:
                main_template = Template(prompt_template)
                
                # 合并所有变量
                render_vars = {
                    **variables,  # 模板预设变量
                    "current_transcript": current_transcript,
                    "history_section": history_section,
                    "requirement_section": requirement_section,
                    "user_requirement": user_requirement or "",  # 添加 user_requirement 变量，用于 smart_prompt
                    "mapping_section": mapping_section,
                    "reference_document_section": reference_document_section,  # 新增：参考文档部分
                    **kwargs  # 其他自定义变量（包括 reference_document）
                }
                
                final_prompt = main_template.render(**render_vars)
                
                # 调试：检查录音内容是否被正确传递
                transcript_length = len(current_transcript) if current_transcript else 0
                logger.info(
                    f"✅ 模板渲染成功 "
                    f"(历史: {'✓' if history_section else '✗'}, "
                    f"需求: {'✓' if requirement_section else '✗'}, "
                    f"参考文档: {'✓' if reference_document_section else '✗'}, "
                    f"录音内容长度: {transcript_length} 字符)"
                )
                
                # 调试：检查最终 prompt 中是否包含录音内容
                if transcript_length > 0:
                    # 检查是否包含 current_transcript 变量（可能是 {{current_transcript}} 或已渲染的内容）
                    if "{{current_transcript}}" in final_prompt:
                        logger.error(f"❌ 错误：最终 prompt 中包含未渲染的变量 {{current_transcript}}！模板渲染可能失败！")
                    elif current_transcript[:100] not in final_prompt:
                        logger.warning(f"⚠️ 警告：最终 prompt 中可能未包含录音内容！")
                        logger.warning(f"   录音内容前100字符: {current_transcript[:100]}")
                        logger.warning(f"   最终 prompt 前500字符: {final_prompt[:500]}")
                    else:
                        logger.info(f"✅ 确认：最终 prompt 中包含录音内容（前100字符匹配）")
                else:
                    logger.warning("⚠️ 警告：录音内容为空！")
                
                return final_prompt
                
            except TemplateError as e:
                logger.error(f"❌ 主模板渲染失败: {e}")
                # 降级：返回不带模板的版本
                return PromptTemplateService._fallback_prompt(
                    current_transcript,
                    history_context,
                    user_requirement
                )
                
        except Exception as e:
            logger.error(f"❌ 模板渲染异常: {e}")
            # 降级：返回简单版本
            return PromptTemplateService._fallback_prompt(
                current_transcript,
                history_context,
                user_requirement
            )
    
    @staticmethod
    def _build_history_content(history_context: Dict) -> str:
        """
        构建历史会议内容文本
        
        Args:
            history_context: 历史会议上下文
        
        Returns:
            格式化的历史内容文本
        """
        mode = history_context.get("mode")
        
        if mode == "retrieval":
            # 检索模式：显示相关片段
            relevant_segments = history_context.get("relevant_segments", [])
            summary = history_context.get("summary", "")
            
            content_parts = []
            
            if summary:
                content_parts.append(f"检索摘要：{summary}")
            
            if relevant_segments:
                content_parts.append(f"\n相关片段（共 {len(relevant_segments)} 条）：")
                for i, seg in enumerate(relevant_segments[:5], 1):  # 最多显示5条
                    meeting_id = seg.get("meeting_id", "未知")
                    speaker = seg.get("speaker", "未知")
                    text = seg.get("text", "")[:150]  # 截断过长文本
                    content_parts.append(
                        f"{i}. [{meeting_id} - {speaker}] {text}..."
                    )
            
            return "\n".join(content_parts)
        
        elif mode == "summary":
            # 总结模式：显示整体总结
            overall_summary = history_context.get("overall_summary", "")
            key_themes = history_context.get("key_themes", [])
            processed_count = history_context.get("processed_count", 0)
            
            content_parts = []
            
            content_parts.append(f"历史会议总结（基于 {processed_count} 个会议）：")
            content_parts.append(overall_summary)
            
            if key_themes:
                content_parts.append(f"\n主要主题：{', '.join(key_themes[:5])}")
            
            return "\n".join(content_parts)
        
        return ""
    
    @staticmethod
    def _fallback_prompt(
        current_transcript: str,
        history_context: Optional[Dict] = None,
        user_requirement: Optional[str] = None
    ) -> str:
        """
        降级提示词（模板渲染失败时使用）
        
        Args:
            current_transcript: 当前会议转录
            history_context: 历史会议上下文
            user_requirement: 用户需求
        
        Returns:
            简单的提示词
        """
        prompt_parts = [
            "请基于以下会议转录生成会议纪要：\n",
            f"【会议转录】\n{current_transcript}\n"
        ]
        
        if history_context:
            prompt_parts.append("\n【历史会议参考】")
            prompt_parts.append("请考虑历史会议背景。\n")
        
        if user_requirement:
            prompt_parts.append(f"\n【用户要求】\n{user_requirement}\n")
        
        prompt_parts.append(
            "\n【输出格式】\n"
            "请输出包含以下部分的会议纪要：\n"
            "1. 会议主题\n"
            "2. 讨论内容\n"
            "3. 决策事项\n"
            "4. 行动项\n"
        )
        
        return "".join(prompt_parts)
    
    @staticmethod
    def parse_template_from_string(template_str: str) -> Optional[Dict[str, Any]]:
        """
        从JSON字符串解析模板配置
        
        Args:
            template_str: JSON格式的模板字符串
        
        Returns:
            模板配置字典，解析失败返回None
        """
        try:
            template_config = json.loads(template_str)
            
            # 验证必需字段
            if "prompt_template" not in template_config:
                logger.error("❌ 模板配置缺少 prompt_template 字段")
                return None
            
            logger.info(
                f"✅ 模板解析成功: {template_config.get('template_name', '未命名')}"
            )
            return template_config
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ 模板JSON解析失败: {e}")
            logger.error(f"   尝试解析的内容（前100字符）: {template_str[:100]}")
            logger.error(f"   💡 提示：如果内容包含反斜杠，请使用双反斜杠（\\\\）或正斜杠（/）")
            return None
    
    @staticmethod
    def get_template_config(
        prompt_template: Optional[str] = None,
        template_id: str = "default",
        strict: bool = False
    ) -> Dict[str, Any]:
        """
        获取模板配置（优先使用自定义模板）
        
        Args:
            prompt_template: 自定义模板（JSON字符串或文档路径）
            template_id: 默认模板ID（或文档路径）
            strict: 严格模式，如果自定义模板解析失败则抛出异常
        
        Returns:
            模板配置字典
        
        Raises:
            ValueError: 严格模式下，自定义模板解析失败时抛出
        """
        # 1. 优先使用自定义模板
        if prompt_template and prompt_template.strip():
            # 清理可能的干扰字符
            cleaned = prompt_template.strip().strip('"').strip("'")
            
            # ⭐ 如果是URL，支持从URL加载模板内容（可为纯文本或文档）
            if cleaned.startswith(("http://", "https://")):
                try:
                    import requests
                    import tempfile
                    import os
                    from app.services.document import document_service

                    logger.info(f"🔗 检测到自定义模板URL: {cleaned}")
                    resp = requests.get(cleaned, timeout=25)
                    resp.raise_for_status()

                    # 根据URL后缀或Content-Type判断是否按文档处理
                    url_path = cleaned.split("?", 1)[0]
                    lower_path = url_path.lower()
                    content_type = resp.headers.get("Content-Type", "").lower()

                    is_doc = lower_path.endswith((".docx", ".pdf", ".txt")) or any(
                        t in content_type
                        for t in [
                            "application/pdf",
                            "application/vnd.openxmlformats-officedocument",
                            "application/msword",
                            "text/plain",
                        ]
                    )

                    template_content = ""
                    if is_doc:
                        # 保存到临时文件再用 document_service 解析
                        suffix = os.path.splitext(url_path)[1] or ".txt"
                        tmp_dir = tempfile.gettempdir()
                        tmp_path = os.path.join(
                            tmp_dir, f"tmpl_{uuid.uuid4().hex}{suffix}"
                        )
                        with open(tmp_path, "wb") as f:
                            f.write(resp.content)
                        try:
                            template_content = document_service.extract_text_from_file(
                                tmp_path
                            )
                        finally:
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                    else:
                        # 直接按文本模板使用
                        template_content = resp.text

                    if template_content and template_content.strip():
                        logger.info(
                            f"✅ 成功从URL加载自定义模板内容，长度: {len(template_content)}"
                        )
                        return {
                            "template_id": "custom_from_url",
                            "template_name": f"URL模板: {cleaned}",
                            "prompt_template": template_content,
                            "variables": {},
                            "dynamic_sections": {},
                        }
                    else:
                        logger.error("❌ URL模板内容为空")
                except Exception as e:
                    logger.error(f"❌ 从URL加载自定义模板失败: {e}")
                    # 继续往下走，尝试其他方式或使用默认模板
            
            # ⭐ 检查是否是文档路径（支持 .docx, .pdf, .txt）
            if cleaned.lower().endswith(('.docx', '.pdf', '.txt')):
                logger.info(f"📂 检测到模板文档路径: {cleaned}")
                
                # 尝试读取文档内容
                import os
                if os.path.exists(cleaned):
                    try:
                        from app.services.document import document_service
                        # ⭐ 使用 extract_html_from_file 保持原模板格式（HTML5）
                        template_html = document_service.extract_html_from_file(cleaned)
                        
                        if template_html and template_html.strip():
                            logger.info(f"✅ 成功读取模板文档（HTML格式），长度: {len(template_html)}")
                            
                            # 从 HTML 中提取纯文本用于检测占位符
                            import re
                            # 移除 HTML 标签，保留文本内容
                            template_text = re.sub(r'<[^>]+>', '', template_html)
                            
                            # ⭐ 智能检测：是否包含占位符（说明是格式模板而非提示词）
                            is_format_template = any([
                                '[请填写' in template_text,
                                '[例如：' in template_text,
                                'XXXX' in template_text,
                                '[填写' in template_text,
                                '【请填写' in template_text,
                                '待补充' in template_text,
                                '待填写' in template_text,
                            ])
                            
                            if is_format_template:
                                logger.info("🎯 检测到格式模板（包含占位符），将作为输出格式要求，使用原HTML格式")
                                # 构建一个智能提示词，让 LLM 根据转录内容填充模板
                                smart_prompt = f"""你是一位专业的会议纪要整理助手。

## 🚨🚨🚨 【第一优先级 - 用户要求】必须100%严格遵守！
{{{{user_requirement}}}}

⚠️⚠️⚠️ **铁律**：
1. **用户的任何要求都必须无条件执行，优先级高于一切（包括格式模板、标准规范等）**
2. **如果用户要求与格式模板冲突，完全以用户要求为准，忽略模板的相应部分**
3. **用户要求什么就生成什么，用户不要什么就完全省略什么**

## 🚨🚨🚨 【核心要求 - 严格基于录音内容】🚨🚨🚨
**绝对禁止添加录音中没有的内容！**
1. **所有内容必须严格来源于下方的【会议录音转录内容】**
2. **如果录音中没有提到某项内容，必须填写"未讨论"或"无"，绝不能编造或推测**
3. **禁止基于常识、历史会议或参考文档来补充录音中没有的内容**
4. **如果模板要求的内容在录音中完全不存在，直接填写"未讨论"，不要保留占位符**
5. **参考文档和历史会议仅用于理解上下文，不能作为生成内容的依据**

## 任务说明
请根据以下**会议录音转录内容**，严格按照**格式模板**的结构和格式生成会议纪要。

## 格式模板（必须严格保持原格式，包括标题、段落结构、标点符号等）
{template_html}

## 会议录音转录内容（这是唯一的内容来源）
{{{{current_transcript}}}}

## 历史会议背景（仅用于理解上下文，不能作为生成内容的依据）
{{{{history_section}}}}

## 参考文档（仅用于理解上下文，不能作为生成内容的依据）
{{{{reference_document_section}}}}

## 基本要求
1. **必须严格根据录音转录内容填充**，不要保留任何占位符（如 `[请填写...]`、`XXXX`、`[例如：...]`、`待补充`等）
2. **所有方括号 `[]` 内的内容都是提示，必须替换为实际内容或"未讨论"**
3. **如果录音中没有提及模板要求的某项内容，必须填写"未讨论"或"无"，绝不能编造**
4. **模板中的Markdown标题(#、##)和所有标点、序号必须100%原样保留**
5. **人名、项目名使用 `<mark class="person">` 和 `<mark class="project">` 标记**
6. **存疑内容使用 `<mark class="uncertain">` 标记**

## 🔥 最后检查（生成前必须确认）
✅ 是否100%严格遵守了用户要求？
✅ 是否所有内容都来源于录音转录内容？
✅ 是否没有添加任何录音中没有的内容？
✅ 如果用户要求与模板冲突，是否以用户要求为准？
✅ 是否用实际内容或"未讨论"替换了所有占位符？
✅ 是否保持了模板的原始格式（标题、段落、标点等）？

请立即生成会议纪要！"""
                                
                                return {
                                    "template_id": "custom_format_template",
                                    "template_name": f"格式模板: {os.path.basename(cleaned)}",
                                    "prompt_template": smart_prompt,
                                    "variables": {},
                                    "dynamic_sections": {}
                                }
                            else:
                                logger.info("📝 检测到提示词模板（无占位符），直接使用HTML格式")
                                # 直接作为提示词使用
                                return {
                                    "template_id": "custom_from_doc",
                                    "template_name": f"文档模板: {os.path.basename(cleaned)}",
                                    "prompt_template": template_html,
                                    "variables": {},
                                    "dynamic_sections": {}
                                }
                        else:
                            logger.error(f"❌ 模板文档内容为空: {cleaned}")
                    except Exception as e:
                        logger.error(f"❌ 读取模板文档失败: {e}")
                else:
                    logger.error(f"❌ 模板文档不存在: {cleaned}")
                
                # 文档读取失败，降级
                logger.warning("⚠️ 文档模板读取失败，降级使用默认模板")
            
            # 检查是否是JSON格式
            elif cleaned.startswith('{') and cleaned.endswith('}'):
                custom_config = PromptTemplateService.parse_template_from_string(cleaned)
                if custom_config:
                    logger.info("📝 使用自定义JSON模板")
                    return custom_config
                else:
                    error_msg = "自定义模板JSON解析失败，请检查JSON格式是否正确"
                    logger.error(f"❌ {error_msg}")
                    if strict:
                        raise ValueError(error_msg)
                    logger.warning("⚠️ 降级使用默认模板")
            else:
                # 既不是文档路径，也不是JSON，可能是纯文本模板
                logger.info("📝 使用纯文本自定义模板")
                return {
                    "template_id": "custom_plain",
                    "template_name": "纯文本自定义模板",
                    "prompt_template": cleaned,
                    "variables": {},
                    "dynamic_sections": {}
                }
        
        # 2. 使用 template_id（也可能是文档路径）
        # 检查 template_id 是否是文档路径
        if template_id and template_id.strip():
            cleaned_tid = template_id.strip().strip('"').strip("'")
            
            # ⭐ template_id 也支持 URL（下载后作为模板使用）
            if cleaned_tid.startswith(("http://", "https://")):
                try:
                    import requests
                    import tempfile
                    import os
                    from app.services.document import document_service

                    logger.info(f"🔗 检测到 template_id 为URL: {cleaned_tid}")
                    resp = requests.get(cleaned_tid, timeout=15)
                    resp.raise_for_status()

                    url_path = cleaned_tid.split("?", 1)[0]
                    lower_path = url_path.lower()
                    content_type = resp.headers.get("Content-Type", "").lower()

                    is_doc = lower_path.endswith((".docx", ".pdf", ".txt")) or any(
                        t in content_type
                        for t in [
                            "application/pdf",
                            "application/vnd.openxmlformats-officedocument",
                            "application/msword",
                            "text/plain",
                        ]
                    )

                    template_content = ""
                    if is_doc:
                        suffix = os.path.splitext(url_path)[1] or ".txt"
                        tmp_dir = tempfile.gettempdir()
                        tmp_path = os.path.join(
                            tmp_dir, f"tmpl_{uuid.uuid4().hex}{suffix}"
                        )
                        with open(tmp_path, "wb") as f:
                            f.write(resp.content)
                        try:
                            # ⭐ 使用 extract_html_from_file 保持原模板格式（HTML5）
                            template_content = document_service.extract_html_from_file(
                                tmp_path
                            )
                        finally:
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass
                    else:
                        template_content = resp.text

                    if template_content and template_content.strip():
                        logger.info(
                            f"✅ 成功从URL加载 template_id 模板内容，长度: {len(template_content)}"
                        )
                        
                        # ⭐ 检测是否是格式模板（包含占位符）
                        import re
                        if is_doc:
                            # 从 HTML 中提取纯文本用于检测占位符
                            template_text = re.sub(r'<[^>]+>', '', template_content)
                        else:
                            template_text = template_content
                        
                        # 使用更稳健的占位符检测逻辑，减少误判：
                        # 只有在出现多个典型占位符时，才认为是“格式模板”，否则当作普通参考文档处理。
                        placeholder_patterns = [
                            '[请填写',
                            '[例如：',
                            'XXXX',
                            '[填写',
                            '【请填写',
                            '待补充',
                            '待填写',
                        ]
                        hit_count = sum(1 for p in placeholder_patterns if p in template_text)
                        is_format_template = hit_count >= 2
                        
                        if is_format_template and is_doc:
                            logger.info("🎯 检测到URL格式模板（包含占位符），将作为输出格式要求，使用原HTML格式")
                            # 构建一个智能提示词，让 LLM 根据转录内容填充“带占位符”的格式模板
                            smart_prompt = f"""你是一位专业的会议纪要整理助手。

## 🚨🚨🚨 【第一优先级 - 用户要求】必须100%严格遵守！
{{{{user_requirement}}}}

⚠️⚠️⚠️ **铁律**：
1. **用户的任何要求都必须无条件执行，优先级高于一切（包括格式模板、标准规范等）**
2. **如果用户要求与格式模板冲突，完全以用户要求为准，忽略模板的相应部分**
3. **用户要求什么就生成什么，用户不要什么就完全省略什么**

## 🚨🚨🚨 【核心要求 - 严格基于录音内容】🚨🚨🚨
**绝对禁止添加录音中没有的内容！**
1. **所有内容必须严格来源于下方的【会议录音转录内容】**
2. **如果录音中没有提到某项内容，必须填写"未讨论"或"无"，绝不能编造或推测**
3. **禁止基于常识、历史会议或参考文档来补充录音中没有的内容**
4. **如果模板要求的内容在录音中完全不存在，直接填写"未讨论"，不要保留占位符**
5. **参考文档和历史会议仅用于理解上下文，不能作为生成内容的依据**

## 任务说明
请根据以下**会议录音转录内容**，严格按照**格式模板**的结构和格式生成会议纪要。

## 格式模板（必须严格保持原格式，包括标题、段落结构、标点符号等）
{template_content}

## 会议录音转录内容（这是唯一的内容来源）
{{{{current_transcript}}}}

## 历史会议背景（仅用于理解上下文，不能作为生成内容的依据）
{{{{history_section}}}}

## 参考文档（仅用于理解上下文，不能作为生成内容的依据）
{{{{reference_document_section}}}}

## 基本要求
1. **必须严格根据录音转录内容填充**，不要保留任何占位符（如 `[请填写...]`、`XXXX`、`[例如：...]`、`待补充`等）
2. **所有方括号 `[]` 内的内容都是提示，必须替换为实际内容或"未讨论"**
3. **如果录音中没有提及模板要求的某项内容，必须填写"未讨论"或"无"，绝不能编造**
4. **模板中的Markdown标题(#、##)和所有标点、序号必须100%原样保留**
5. **人名、项目名使用 `<mark class="person">` 和 `<mark class="project">` 标记**
6. **存疑内容使用 `<mark class="uncertain">` 标记**

## 🔥 最后检查（生成前必须确认）
✅ 是否100%严格遵守了用户要求？
✅ 是否所有内容都来源于录音转录内容？
✅ 是否没有添加任何录音中没有的内容？
✅ 如果用户要求与模板冲突，是否以用户要求为准？
✅ 是否用实际内容或"未讨论"替换了所有占位符？
✅ 是否保持了模板的原始格式（标题、段落、标点等）？

请立即生成会议纪要！"""
                            
                            return {
                                "template_id": "custom_format_template_url",
                                "template_name": f"URL格式模板: {os.path.basename(url_path)}",
                                "prompt_template": smart_prompt,
                                "variables": {},
                                "dynamic_sections": {}
                            }
                        else:
                            # ⭐ 无占位符的 URL 模板：将其视为“格式参考模板”，用实际会议内容填充，不照抄示例文字
                            logger.info("📝 检测到URL提示词/格式模板（无占位符），将作为格式参考使用，并基于转录内容进行填充")
                            smart_prompt = f"""你是一位专业的会议纪要整理助手。

## 🚨🚨🚨 【第一优先级 - 用户要求】必须100%严格遵守！
{{{{user_requirement}}}}

⚠️⚠️⚠️ **核心原则：内容优先，格式从模板学习**
1. 下方给出了一份 HTML 格式的“会议纪要模板示例”，**只用于学习其版式结构、段落顺序、层级关系和字段名称**。
2. 模板中的示例文字（如“XXXX年XX月XX日”、“某某项目”等）**一律视为占位示例，绝对禁止照抄或当作真实内容输出**。
3. 你必须完全基于【会议内容】（录音转写 / 文本）来填充每个部分，**不得编造录音/文本中不存在的事实**。
4. 如果模板中的某个栏目在会议内容中完全未提及，请在对应位置自然说明“本次会议未明确讨论该项内容”或“暂无相关信息”，而不是留下占位符。

## 输出格式模板示例（仅作版式参考，示例文字不得照抄）
以下 HTML 仅用于说明“应该采用怎样的标题层级和段落结构”，其中的具体文字内容全部是示例：

{template_content}

## 会议内容（这是唯一的信息来源，可以来自录音转写、rebuild 或 text_content）
请严格基于下方内容，提炼出会议纪要并填入上述格式结构中：

{{{{current_transcript}}}}

## 历史会议背景（仅用于理解上下文，不能作为生成内容的唯一依据）
{{{{history_section}}}}

## 参考文档（仅用于理解上下文，不能作为生成内容的唯一依据）
{{{{reference_document_section}}}}

## 输出要求
1. 输出格式必须保持与“输出格式模板示例”的整体结构一致（标题层级、主要栏目名称）。
2. 所有内容必须能够在【会议内容】或参考信息中找到依据，禁止凭空捏造。
3. 对于未在会议中明确讨论的部分，请自然说明“本次会议未明确讨论该项内容”或“暂无相关信息”，不要使用“录音未明确”等表述。
4. 输出内容直接使用 HTML5 格式（支持 <p>、<h1>-<h3>、<ul>/<ol>、<li> 等），不要再包含占位符。"""

                            return {
                                "template_id": "custom_format_template_url_plain",
                                "template_name": f"URL格式参考模板: {cleaned_tid}",
                                "prompt_template": smart_prompt,
                                "variables": {},
                                "dynamic_sections": {}
                            }
                    else:
                        logger.error("❌ template_id URL 模板内容为空")
                except Exception as e:
                    logger.error(f"❌ 从 template_id URL 加载模板失败: {e}")
                    # 失败后继续按原有逻辑处理（本地路径或默认模板）

            if cleaned_tid.lower().endswith(('.docx', '.pdf', '.txt')):
                logger.info(f"📂 检测到template_id是文档路径: {cleaned_tid}")
                
                import os
                if os.path.exists(cleaned_tid):
                    try:
                        from app.services.document import document_service
                        # ⭐ 使用 extract_html_from_file 保持原模板格式（HTML5）
                        template_html = document_service.extract_html_from_file(cleaned_tid)
                        
                        if template_html and template_html.strip():
                            logger.info(f"✅ 成功读取模板文档（HTML格式），长度: {len(template_html)}")
                            
                            # 从 HTML 中提取纯文本用于检测占位符
                            import re
                            # 移除 HTML 标签，保留文本内容
                            template_text = re.sub(r'<[^>]+>', '', template_html)
                            
                            # ⭐ 智能检测：是否包含占位符（说明是格式模板而非提示词）
                            is_format_template = any([
                                '[请填写' in template_text,
                                '[例如：' in template_text,
                                'XXXX' in template_text,
                                '[填写' in template_text,
                                '【请填写' in template_text,
                                '待补充' in template_text,
                                '待填写' in template_text,
                            ])
                            
                            if is_format_template:
                                logger.info("🎯 检测到格式模板（包含占位符），将作为输出格式要求，使用原HTML格式")
                                # 构建一个智能提示词，让 LLM 根据转录内容填充模板
                                smart_prompt = f"""你是一位专业的会议纪要整理助手。

## 🚨🚨🚨 【第一优先级 - 用户要求】必须100%严格遵守！
{{{{user_requirement}}}}

⚠️⚠️⚠️ **铁律**：
1. **用户的任何要求都必须无条件执行，优先级高于一切（包括格式模板、标准规范等）**
2. **如果用户要求与格式模板冲突，完全以用户要求为准，忽略模板的相应部分**
3. **用户要求什么就生成什么，用户不要什么就完全省略什么**

## 🚨🚨🚨 【核心要求 - 严格基于录音内容】🚨🚨🚨
**绝对禁止添加录音中没有的内容！**
1. **所有内容必须严格来源于下方的【会议录音转录内容】**
2. **如果录音中没有提到某项内容，必须填写"未讨论"或"无"，绝不能编造或推测**
3. **禁止基于常识、历史会议或参考文档来补充录音中没有的内容**
4. **如果模板要求的内容在录音中完全不存在，直接填写"未讨论"，不要保留占位符**
5. **参考文档和历史会议仅用于理解上下文，不能作为生成内容的依据**

## 任务说明
请根据以下**会议录音转录内容**，严格按照**格式模板**的结构和格式生成会议纪要。

## 格式模板（必须严格保持原格式，包括标题、段落结构、标点符号等）
{template_html}

## 会议录音转录内容（这是唯一的内容来源）
{{{{current_transcript}}}}

## 历史会议背景（仅用于理解上下文，不能作为生成内容的依据）
{{{{history_section}}}}

## 参考文档（仅用于理解上下文，不能作为生成内容的依据）
{{{{reference_document_section}}}}

## 基本要求
1. **必须严格根据录音转录内容填充**，不要保留任何占位符（如 `[请填写...]`、`XXXX`、`[例如：...]`、`待补充`等）
2. **所有方括号 `[]` 内的内容都是提示，必须替换为实际内容或"未讨论"**
3. **如果录音中没有提及模板要求的某项内容，必须填写"未讨论"或"无"，绝不能编造**
4. **模板中的Markdown标题(#、##)和所有标点、序号必须100%原样保留**
5. **人名、项目名使用 `<mark class="person">` 和 `<mark class="project">` 标记**
6. **存疑内容使用 `<mark class="uncertain">` 标记**

## 🔥 最后检查（生成前必须确认）
✅ 是否100%严格遵守了用户要求？
✅ 是否所有内容都来源于录音转录内容？
✅ 是否没有添加任何录音中没有的内容？
✅ 如果用户要求与模板冲突，是否以用户要求为准？
✅ 是否用实际内容或"未讨论"替换了所有占位符？
✅ 是否保持了模板的原始格式（标题、段落、标点等）？

请立即生成会议纪要！"""
                                
                                return {
                                    "template_id": "custom_format_template",
                                    "template_name": f"格式模板: {os.path.basename(cleaned_tid)}",
                                    "prompt_template": smart_prompt,
                                    "variables": {},
                                    "dynamic_sections": {}
                                }
                            else:
                                # 无占位符的本地文档模板：同样作为“格式参考”，基于会议内容填充
                                logger.info("📝 检测到文档提示词/格式模板（无占位符），将作为格式参考使用，并基于转录内容进行填充")
                                smart_prompt = f"""你是一位专业的会议纪要整理助手。

## 🚨🚨🚨 【第一优先级 - 用户要求】必须100%严格遵守！
{{{{user_requirement}}}}

⚠️⚠️⚠️ **核心原则：内容优先，格式从模板学习**
1. 下方给出了一份 HTML 格式的“会议纪要模板示例”，**只用于学习其版式结构、段落顺序、层级关系和字段名称**。
2. 模板中的示例文字（如“XXXX年XX月XX日”、“某某项目”等）**一律视为占位示例，绝对禁止照抄或当作真实内容输出**。
3. 你必须完全基于【会议内容】（录音转写 / 文本）来填充每个部分，**不得编造录音/文本中不存在的事实**。
4. 如果模板中的某个栏目在会议内容中完全未提及，请在对应位置自然说明“本次会议未明确讨论该项内容”或“暂无相关信息”，而不是留下占位符。

## 输出格式模板示例（仅作版式参考，示例文字不得照抄）
以下 HTML 仅用于说明“应该采用怎样的标题层级和段落结构”，其中的具体文字内容全部是示例：

{template_html}

## 会议内容（这是唯一的信息来源，可以来自录音转写、rebuild 或 text_content）
请严格基于下方内容，提炼出会议纪要并填入上述格式结构中：

{{{{current_transcript}}}}

## 历史会议背景（仅用于理解上下文，不能作为生成内容的唯一依据）
{{{{history_section}}}}

## 参考文档（仅用于理解上下文，不能作为生成内容的唯一依据）
{{{{reference_document_section}}}}

## 输出要求
1. 输出格式必须保持与“输出格式模板示例”的整体结构一致（标题层级、主要栏目名称）。
2. 所有内容必须能够在【会议内容】或参考信息中找到依据，禁止凭空捏造。
3. 对于未在会议中明确讨论的部分，请自然说明“本次会议未明确讨论该项内容”或“暂无相关信息”，不要使用“录音未明确”等表述。
4. 输出内容直接使用 HTML5 格式（支持 <p>、<h1>-<h3>、<ul>/<ol>、<li> 等），不要再包含占位符。"""

                                return {
                                    "template_id": "custom_from_doc_format_ref",
                                    "template_name": f"文档格式参考模板: {os.path.basename(cleaned_tid)}",
                                    "prompt_template": smart_prompt,
                                    "variables": {},
                                    "dynamic_sections": {}
                                }
                    except Exception as e:
                        logger.error(f"❌ 读取模板文档失败: {e}")
        
        # 3. 使用默认模板
        template_config = get_default_template(template_id)
        logger.info(f"📝 使用默认模板: {template_id}")
        return template_config


# 创建单例实例
prompt_template_service = PromptTemplateService()
