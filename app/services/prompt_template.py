"""
动态提示词模板渲染服务
支持Jinja2模板语法和动态变量替换
"""
from typing import Dict, Any, Optional
from jinja2 import Template, TemplateError
import json
import os
from pathlib import Path

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
            mapping_parts = ["【名称标准化映射表】"]
            mapping_parts.append("⚠️ 重要：在生成会议纪要时，请将以下口语化表达替换为标准名称：\n")
            
            for category, mapping_dict in mappings.items():
                if mapping_dict:
                    mapping_parts.append(f"**{category}映射**：")
                    for oral, standard in mapping_dict.items():
                        mapping_parts.append(f"  • \"{oral}\" → \"{standard}\"")
                    mapping_parts.append("")
            
            mapping_parts.append("📝 规则说明：")
            mapping_parts.append("1. 如果转录文本中出现左侧的口语化表达，请在纪要中使用右侧的标准名称")
            mapping_parts.append("2. 第一次出现时使用标准全称，后续可适当使用简称")
            mapping_parts.append("3. 在人名后建议加上职位信息（如果转录中有提及）")
            mapping_parts.append("4. 保持专业性和一致性\n")
            
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
            
            # === 渲染最终 Prompt ===
            try:
                main_template = Template(prompt_template)
                
                # 合并所有变量
                render_vars = {
                    **variables,  # 模板预设变量
                    "current_transcript": current_transcript,
                    "history_section": history_section,
                    "requirement_section": requirement_section,
                    "mapping_section": mapping_section,
                    **kwargs  # 其他自定义变量
                }
                
                final_prompt = main_template.render(**render_vars)
                
                logger.info(
                    f"✅ 模板渲染成功 "
                    f"(历史: {'✓' if history_section else '✗'}, "
                    f"需求: {'✓' if requirement_section else '✗'})"
                )
                
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
            
            # ⭐ 检查是否是文档路径（支持 .docx, .pdf, .txt）
            if cleaned.lower().endswith(('.docx', '.pdf', '.txt')):
                logger.info(f"📂 检测到模板文档路径: {cleaned}")
                
                # 尝试读取文档内容
                import os
                if os.path.exists(cleaned):
                    try:
                        from app.services.document import document_service
                        template_content = document_service.extract_text_from_file(cleaned)
                        
                        if template_content and template_content.strip():
                            logger.info(f"✅ 成功读取模板文档，长度: {len(template_content)}")
                            
                            # ⭐ 智能检测：是否包含占位符（说明是格式模板而非提示词）
                            is_format_template = any([
                                '[请填写' in template_content,
                                '[例如：' in template_content,
                                'XXXX' in template_content,
                                '[填写' in template_content,
                                '【请填写' in template_content,
                            ])
                            
                            if is_format_template:
                                logger.info("🎯 检测到格式模板（包含占位符），将作为输出格式要求")
                                # 构建一个智能提示词，让 LLM 根据转录内容填充模板
                                smart_prompt = f"""你是一位专业的会议纪要整理助手。

## 任务说明
请根据以下**会议录音转录内容**，严格按照**指定格式模板**生成会议纪要。

## 重要要求
1. **必须根据实际会议内容填充**，不要保留任何占位符（如 `[请填写...]`、`XXXX`、`[例如：...]`）
2. **所有方括号 `[]` 内的内容都是提示，必须替换为实际内容**
3. 如果会议中没有提及某项内容，填写"未讨论"或"无"，不要留空或保留占位符
4. 时间格式使用实际时间（从转录内容推断或使用当前时间）
5. 人名、项目名使用 `<mark class="person">` 和 `<mark class="project">` 标记
6. 存疑内容使用 `<mark class="uncertain">` 标记

## 指定格式模板
{template_content}

## 会议录音转录内容
{{{{current_transcript}}}}

## 历史会议背景（如有）
{{{{history_context}}}}

## 用户特殊要求（如有）
{{{{user_requirement}}}}

请严格按照上述格式模板生成完整的会议纪要，确保所有占位符都被实际内容替换！"""
                                
                                return {
                                    "template_id": "custom_format_template",
                                    "template_name": f"格式模板: {os.path.basename(cleaned)}",
                                    "prompt_template": smart_prompt,
                                    "variables": {},
                                    "dynamic_sections": {}
                                }
                            else:
                                logger.info("📝 检测到提示词模板（无占位符），直接使用")
                                # 直接作为提示词使用
                                return {
                                    "template_id": "custom_from_doc",
                                    "template_name": f"文档模板: {os.path.basename(cleaned)}",
                                    "prompt_template": template_content,
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
            
            if cleaned_tid.lower().endswith(('.docx', '.pdf', '.txt')):
                logger.info(f"📂 检测到template_id是文档路径: {cleaned_tid}")
                
                import os
                if os.path.exists(cleaned_tid):
                    try:
                        from app.services.document import document_service
                        template_content = document_service.extract_text_from_file(cleaned_tid)
                        
                        if template_content and template_content.strip():
                            logger.info(f"✅ 成功读取模板文档，长度: {len(template_content)}")
                            
                            # ⭐ 智能检测：是否包含占位符（说明是格式模板而非提示词）
                            is_format_template = any([
                                '[请填写' in template_content,
                                '[例如：' in template_content,
                                'XXXX' in template_content,
                                '[填写' in template_content,
                                '【请填写' in template_content,
                            ])
                            
                            if is_format_template:
                                logger.info("🎯 检测到格式模板（包含占位符），将作为输出格式要求")
                                # 构建一个智能提示词，让 LLM 根据转录内容填充模板
                                smart_prompt = f"""你是一位专业的会议纪要整理助手。

## 任务说明
请根据以下**会议录音转录内容**，严格按照**指定格式模板**生成会议纪要。

## 重要要求
1. **必须根据实际会议内容填充**，不要保留任何占位符（如 `[请填写...]`、`XXXX`、`[例如：...]`）
2. **所有方括号 `[]` 内的内容都是提示，必须替换为实际内容**
3. 如果会议中没有提及某项内容，填写"未讨论"或"无"，不要留空或保留占位符
4. 时间格式使用实际时间（从转录内容推断或使用当前时间）
5. 人名、项目名使用 `<mark class="person">` 和 `<mark class="project">` 标记
6. 存疑内容使用 `<mark class="uncertain">` 标记

## 指定格式模板
{template_content}

## 会议录音转录内容
{{{{current_transcript}}}}

## 历史会议背景（如有）
{{{{history_context}}}}

## 用户特殊要求（如有）
{{{{user_requirement}}}}

请严格按照上述格式模板生成完整的会议纪要，确保所有占位符都被实际内容替换！"""
                                
                                return {
                                    "template_id": "custom_format_template",
                                    "template_name": f"格式模板: {os.path.basename(cleaned_tid)}",
                                    "prompt_template": smart_prompt,
                                    "variables": {},
                                    "dynamic_sections": {}
                                }
                            else:
                                logger.info("📝 检测到提示词模板（无占位符），直接使用")
                                return {
                                    "template_id": "custom_from_doc",
                                    "template_name": f"文档模板: {os.path.basename(cleaned_tid)}",
                                    "prompt_template": template_content,
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
