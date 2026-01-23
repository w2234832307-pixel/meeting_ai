"""
默认提示词模板管理
支持动态变量替换和模板继承
"""
from typing import Dict, Any

# ============================================
# 默认模板定义
# ============================================

DEFAULT_TEMPLATES = {
    "default": {
        "template_id": "default",
        "template_name": "标准会议纪要",
        "description": "适用于一般会议的标准纪要格式",
        "version": "1.0",
        "prompt_template": """你是一个专业的会议纪要助手。请基于以下信息生成结构化的会议纪要。

【当前会议转录】
{current_transcript}

{history_section}

{requirement_section}

【输出要求】
1. 提取会议基本信息（时间、参与人员、主题）
2. 总结会议内容（150字以内）
3. 列出讨论议题（分点列出，每点100字以内）
4. 提取重要决策和行动项
5. 列出后续跟进事项

【输出格式】
请以JSON格式输出，包含以下字段：
{{
  "meeting_info": {{
    "date": "会议时间",
    "participants": ["参与人员列表"],
    "topic": "会议主题"
  }},
  "summary": "会议摘要",
  "topics": [
    {{"title": "议题1", "content": "讨论内容"}},
    {{"title": "议题2", "content": "讨论内容"}}
  ],
  "decisions": ["决策1", "决策2"],
  "action_items": [
    {{"task": "任务描述", "owner": "负责人", "deadline": "截止日期"}}
  ],
  "follow_ups": ["跟进事项1", "跟进事项2"]
}}
""",
        "variables": {},
        "dynamic_sections": {
            "history_section": """
【历史会议参考】
{history_content}

请在生成纪要时考虑历史背景和延续性。
""",
            "requirement_section": """
【用户特别要求】
{user_requirement}

⚠️ 请在生成纪要时重点关注用户要求的内容。
"""
        }
    },
    
    "simple": {
        "template_id": "simple",
        "template_name": "简洁版纪要",
        "description": "简洁快速的纪要格式",
        "version": "1.0",
        "prompt_template": """请用简洁的语言总结以下会议内容：

【会议转录】
{current_transcript}

{history_section}

{requirement_section}

【输出要求】
只输出以下内容（Markdown格式）：
1. 会议主题（一句话）
2. 3个关键讨论点（分点列出）
3. 决策事项（如果有）
4. 行动项（如果有）

保持简洁，总字数不超过300字。
""",
        "variables": {},
        "dynamic_sections": {
            "history_section": """
【历史会议参考】
{history_content}
""",
            "requirement_section": """
【用户要求】
{user_requirement}
"""
        }
    },
    
    "detailed": {
        "template_id": "detailed",
        "template_name": "详细版纪要",
        "description": "包含完整细节的详细纪要",
        "version": "1.0",
        "prompt_template": """你是一个资深会议记录专家。请基于以下信息生成详细的会议纪要。

【当前会议转录】
{current_transcript}

{history_section}

{requirement_section}

【输出要求】
1. 会议基本信息
   - 时间、地点、参与人员、会议主题
   
2. 会议背景
   - 会议目的和背景说明
   
3. 详细讨论内容
   - 每个议题的详细讨论过程
   - 各参与人的主要观点
   - 支持和反对的理由
   
4. 决策事项
   - 明确的决策内容
   - 决策的理由和依据
   
5. 行动项
   - 具体任务、负责人、截止日期
   
6. 遗留问题
   - 未解决的问题
   - 后续需要跟进的事项

【输出格式】
使用Markdown格式，结构清晰，层次分明。
""",
        "variables": {},
        "dynamic_sections": {
            "history_section": """
【历史会议背景】
{history_content}

请在纪要中体现会议的延续性和发展脉络。
""",
            "requirement_section": """
【特别关注点】
{user_requirement}

请在纪要中重点突出这些内容。
"""
        }
    },
    
    "financial": {
        "template_id": "financial",
        "template_name": "财务会议纪要",
        "description": "适用于财务相关会议",
        "version": "1.0",
        "prompt_template": """你是一个财务会议记录专家。请基于以下信息生成财务会议纪要。

【当前会议转录】
{current_transcript}

{history_section}

{requirement_section}

【输出要求】
1. 会议基本信息（时间、参与人、主题）

2. 财务数据摘要
   - 提取所有涉及的金额、预算、成本等数据
   - 数据对比（同比、环比）
   
3. 财务议题讨论
   - 预算审批
   - 成本控制
   - 投资决策
   - 风险评估
   
4. 财务决策
   - 批准的预算或支出
   - 财务政策调整
   
5. 风险提示
   - 财务风险点
   - 合规性问题
   
6. 行动项
   - 财务相关的后续任务

【输出格式】
使用结构化的Markdown格式，所有金额用**加粗**标注。
""",
        "variables": {},
        "dynamic_sections": {
            "history_section": """
【历史财务数据参考】
{history_content}

请在分析时对比历史数据趋势。
""",
            "requirement_section": """
【财务关注重点】
{user_requirement}
"""
        }
    }
}


def get_default_template(template_id: str = "default") -> Dict[str, Any]:
    """
    获取默认模板配置
    
    Args:
        template_id: 模板ID
    
    Returns:
        模板配置字典
    """
    return DEFAULT_TEMPLATES.get(template_id, DEFAULT_TEMPLATES["default"]).copy()


def list_templates() -> Dict[str, Dict[str, str]]:
    """
    列出所有可用的模板
    
    Returns:
        模板列表（ID -> {name, description}）
    """
    return {
        tid: {
            "name": tpl["template_name"],
            "description": tpl["description"],
            "version": tpl["version"]
        }
        for tid, tpl in DEFAULT_TEMPLATES.items()
    }
