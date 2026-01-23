"""
动态提示词模板渲染服务
支持Jinja2模板语法和动态变量替换
"""
from typing import Dict, Any, Optional
from jinja2 import Template, TemplateError
import json

from app.core.logger import logger
from app.prompts.templates import get_default_template


class PromptTemplateService:
    """提示词模板渲染服务"""
    
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
            
            # === 渲染最终 Prompt ===
            try:
                main_template = Template(prompt_template)
                
                # 合并所有变量
                render_vars = {
                    **variables,  # 模板预设变量
                    "current_transcript": current_transcript,
                    "history_section": history_section,
                    "requirement_section": requirement_section,
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
            return None
    
    @staticmethod
    def get_template_config(
        prompt_template: Optional[str] = None,
        template_id: str = "default"
    ) -> Dict[str, Any]:
        """
        获取模板配置（优先使用自定义模板）
        
        Args:
            prompt_template: 自定义模板JSON字符串
            template_id: 默认模板ID
        
        Returns:
            模板配置字典
        """
        # 1. 优先使用自定义模板
        if prompt_template:
            custom_config = PromptTemplateService.parse_template_from_string(
                prompt_template
            )
            if custom_config:
                logger.info("📝 使用自定义模板")
                return custom_config
            else:
                logger.warning("⚠️ 自定义模板解析失败，使用默认模板")
        
        # 2. 使用默认模板
        template_config = get_default_template(template_id)
        logger.info(f"📝 使用默认模板: {template_id}")
        return template_config


# 创建单例实例
prompt_template_service = PromptTemplateService()
