"""
本地LLM服务（支持Qwen3-14b等本地部署模型）
使用OpenAI兼容接口调用本地模型
"""
import json
import re
from typing import Dict, Optional
from openai import OpenAI

from app.core.config import settings
from app.core.logger import logger
from app.core.exceptions import LLMServiceException


def remove_thinking_tags(text: str) -> str:
    """
    移除LLM输出中的思考过程标签
    支持多种格式：
    1. <think>...</think>
    2. <p>...思考内容...</p>...<h3>会议纪要</h3>
    3. HTML嵌套的各种变体
    4. <p>语种：中文<br /></think></p> (跨行残留)
    """
    if not text:
        return text
    
    original_length = len(text)
    
    # === 策略1: 移除标准 <think> 标签（包括跨行） ===
    text = re.sub(r'<think>[\s\S]*?</think>', '', text, flags=re.IGNORECASE)
    
    # === 策略2: 移除残留的 </think> 标签及其所在的段落 ===
    # 匹配包含 </think> 的整个 <p> 标签（包括跨行、包含 <br />）
    text = re.sub(r'<p>[\s\S]*?</think>[\s\S]*?</p>', '', text, flags=re.IGNORECASE)
    
    # === 策略3: 移除开头的思考内容 - 从开头到第一个 Markdown 标题 ===
    # 检测是否以 <p> 或空白开头，且后面有 Markdown 标题（###、##、#）
    if re.search(r'^[\s\S]*?#{1,3}\s', text):
        # 查找第一个标题的位置
        match = re.search(r'#{1,3}\s', text)
        if match:
            # 检查标题之前的内容是否包含思考关键词
            before_title = text[:match.start()]
            thinking_indicators = ['语种', '好的', '首先', '接下来', '需要', '思考', '<p>', '</think>']
            if any(indicator in before_title for indicator in thinking_indicators):
                text = text[match.start():]
                logger.info("🧹 检测到开头的思考内容，已移除")
    
    # === 策略4: 移除包含思考关键词的 <p> 段落 ===
    thinking_patterns = [
        r'<p>[\s\S]*?语种[\s\S]*?</p>',  # 语种标识
        r'<p>[\s\S]*?好的，我.*?</p>',
        r'<p>[\s\S]*?首先.*?</p>',
        r'<p>[\s\S]*?接下来.*?</p>',
        r'<p>[\s\S]*?需要注意.*?</p>',
        r'<p>[\s\S]*?最后，需要.*?</p>',
    ]
    for pattern in thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # === 清理残留 ===
    # 移除空的 <p> 标签
    text = re.sub(r'<p>\s*</p>', '', text)
    
    # 移除开头的无用标签和空白（保留真正的HTML起始标签，如 <p>）
    # 同样不删除 '<' 和 '>'，避免破坏正常的 HTML 结构
    text = re.sub(r'^[\s"\n]*', '', text)
    
    # 移除多余的空白行
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 去除开头和结尾的空白和引号
    text = text.strip().strip('"').strip()
    
    removed_chars = original_length - len(text)
    if removed_chars > 0:
        logger.info(f"🧹 已清理思考内容: 移除 {removed_chars} 字符")
    
    return text


def add_highlighting(text: str) -> str:
    """
    为会议纪要添加高亮标记
    - 人名：用 <mark class="person">...</mark> 包裹
    - 日期/时间：用 <mark class="date">...</mark> 包裹
    - 存疑内容：用 <mark class="uncertain">...</mark> 包裹
    - 项目名：用 <mark class="project">...</mark> 包裹
    
    Args:
        text: Markdown格式的会议纪要
    
    Returns:
        添加了高亮标记的文本
    """
    if not text:
        return text

    # 如果内容本身是合法 JSON（例如说话人摘要 JSON），为避免破坏结构，直接跳过高亮
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return text
        except Exception:
            # 不是纯 JSON，再继续下面的高亮逻辑
            pass
    
    # === 1. 高亮人名 ===
    # 1.1 带引号的人名：匹配中文引号、英文引号包裹的1-4个字的中文人名
    text = re.sub(
        r'[""]([一-龥]{1,4})[""]',
        r'<mark class="person">\1</mark>',
        text
    )
    
    # 1.2 <strong> 标签中的人名（LLM常用格式）
    # 匹配 <strong>唐玉</strong>、<strong>子波</strong>、<strong>李</strong> 等格式
    # 扩展到1-10个字，支持复姓和更长的名字
    text = re.sub(
        r'<strong>([一-龥]{1,10})</strong>',
        r'<mark class="person">\1</mark>',
        text
    )
    
    # 1.3 **Markdown加粗**中的人名
    # 匹配 **唐玉**、**子波**、**李** 等格式
    text = re.sub(
        r'\*\*([一-龥]{1,10})\*\*',
        r'<mark class="person">\1</mark>',
        text
    )
    
    # === 2. 高亮项目名/产品名 ===
    # 通用项目名模式（不依赖特定名称）
    project_patterns = [
        # 大写字母项目名：OMC、ONC、FSU、AI等（2-10个连续大写字母）
        (r'\b([A-Z]{2,10})\b', r'<mark class="project">\1</mark>'),
        # 带"项目"、"产品"、"系统"、"平台"、"工具"、"服务"后缀的名称
        (r'([一-龥0-9A-Za-z]{2,15}(?:项目|产品|系统|平台|工具|服务|计划|方案|库))', r'<mark class="project">\1</mark>'),
    ]
    for pattern, replacement in project_patterns:
        text = re.sub(pattern, replacement, text)
    
    # === 3. 高亮日期和时间 ===
    date_patterns = [
        # 周X
        (r'(周[一二三四五六日天])', r'<mark class="date">\1</mark>'),
        # 今天、明天、后天、昨天
        (r'(今天|明天|后天|昨天|前天)', r'<mark class="date">\1</mark>'),
        # 本周、下周、上周
        (r'(本周|下周|上周|这周|上上周)', r'<mark class="date">\1</mark>'),
        # 本月、下月、上月
        (r'(本月|下月|上月|这个月)', r'<mark class="date">\1</mark>'),
        # X月X日
        (r'(\d{1,2}月\d{1,2}日)', r'<mark class="date">\1</mark>'),
        # YYYY-MM-DD、YYYY/MM/DD
        (r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', r'<mark class="date">\1</mark>'),
        # 时间点：如"周五"、"周二至周三"
        (r'(周[一二三四五六日天]至周[一二三四五六日天])', r'<mark class="date">\1</mark>'),
    ]
    for pattern, replacement in date_patterns:
        text = re.sub(pattern, replacement, text)
    
    # === 4. 高亮存疑内容（基于ASR低置信度标记）===
    # 注意：这里高亮的是LLM已经标记为【存疑】的内容
    # 如果ASR识别有低置信度，会用特殊标记包裹，如：【存疑：某个词】
    uncertain_patterns = [
        # ASR低置信度标记（由LLM生成的存疑标记）
        (r'【存疑[：:]\s*([^】]+)】', r'<mark class="uncertain">\1</mark>'),
        (r'\[存疑[：:]\s*([^\]]+)\]', r'<mark class="uncertain">\1</mark>'),
    ]
    for pattern, replacement in uncertain_patterns:
        text = re.sub(pattern, replacement, text)
    
    logger.info("✨ 已添加高亮标记（人名、项目名、日期、ASR存疑内容）")
    return text


class LocalLLMService:
    """本地LLM服务类（Qwen3-14b等）"""
    
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None, test_on_init: bool = None):
        """
        初始化本地LLM客户端
        
        Args:
            api_key: API密钥（如果为None，使用配置文件）
            base_url: API地址（如果为None，使用配置文件）
            model_name: 模型名称（如果为None，使用配置文件）
            test_on_init: 是否初始化时测试连接（如果为None，使用配置文件）
        """
        try:
            logger.info("🚀 正在初始化本地LLM服务...")
            
            # 使用传入参数或配置文件值
            self.api_key = api_key or settings.LOCAL_LLM_API_KEY or "dummy-key"
            self.base_url = base_url or settings.LOCAL_LLM_BASE_URL
            self.model = model_name or settings.LOCAL_LLM_MODEL_NAME
            self.test_on_init = test_on_init if test_on_init is not None else settings.LOCAL_LLM_TEST_ON_INIT
            
            # 初始化 OpenAI 兼容客户端（指向本地服务）
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=settings.LLM_TIMEOUT
            )
            
            # 测试连接
            if self.test_on_init:
                self._test_connection()
            
            logger.info(f"✅ 本地LLM服务初始化成功 (模型: {self.model}, URL: {self.base_url})")
            
        except Exception as e:
            logger.error(f"❌ 本地LLM服务初始化失败: {e}")
            raise LLMServiceException(f"本地LLM初始化失败: {str(e)}")
    
    def _test_connection(self):
        """测试本地LLM服务连接"""
        try:
            logger.info("🔍 测试本地LLM服务连接...")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "你好"}],
                max_tokens=10,
                temperature=0.1
            )
            if response and response.choices:
                logger.info("✅ 本地LLM服务连接测试成功")
            else:
                raise LLMServiceException("本地LLM服务响应异常")
        except Exception as e:
            logger.warning(f"⚠️ 本地LLM服务连接测试失败: {e}")
            # 不抛出异常，允许后续使用时再报错
    
    def judge_rag(self, raw_text: str, template_id: str) -> dict:
        """
        判断是否需要RAG检索，并提取搜索关键词
        
        Args:
            raw_text: 原始文本
            template_id: 模板ID（暂未使用）
        
        Returns:
            包含 need_rag 和 search_query 的字典
        """
        logger.info("🧠 本地LLM 正在分析 RAG 意图并提取关键词...")
        
        prompt = f"""
你是一个专业的会议秘书。请分析以下会议记录（ASR识别文本），判断是否需要检索历史知识库来辅助生成纪要。

【会议内容】：
"{raw_text[:2000]}..." 

【判断标准】：
如果文中出现了模糊指代（如"上次说的"、"那个项目"）或提到具体的历史问题、技术名词，则需要检索。

请严格返回 JSON 格式：
{{
    "need_rag": true,
    "search_query": "提取出的核心搜索关键词，用空格分隔，不要包含废话" 
}}
或者
{{
    "need_rag": false,
    "search_query": ""
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                response_format={"type": "json_object"} if settings.LOCAL_LLM_SUPPORT_JSON_MODE else None
            )
            content = response.choices[0].message.content
            
            # 解析JSON（即使不支持json_object模式，也尝试从文本中提取）
            result = self._extract_json_from_text(content)
            
            logger.info(f"✅ RAG分析完成: need_rag={result.get('need_rag', False)}")
            return result
            
        except Exception as e:
            logger.error(f"❌ RAG 分析失败: {e}")
            return {"need_rag": False, "search_query": ""}
    
    def generate_markdown(self, raw_text: str, context: str = "", template_id: str = "default") -> str:
        """
        根据模板生成结构化数据
        
        Args:
            raw_text: 原始文本
            context: RAG检索到的历史上下文（如果有）
            template_id: 模板ID，用于选择不同的生成模板
        
        Returns:
            根据模板生成的结构化数据（Markdown格式）
        """
        logger.info(f"🧠 本地LLM 正在根据模板 '{template_id}' 生成结构化数据...")
        
        # 根据模板ID选择不同的模板
        template = self._get_template(template_id)
        
        # 组装 Prompt
        system_prompt = template.get("system_prompt", "你是一个专业的高级秘书，负责将语音识别的文本整理成结构清晰的数据。")
        
        user_input = template.get("user_prompt_template", "").format(
            context=context if context else "无",
            raw_text=raw_text
        )
        
        if not user_input:
            # 默认模板
            user_input = f"""
请根据以下内容生成会议纪要。

【参考历史信息】：
{context if context else "无"}

【原始语音文本】：
{raw_text}

【要求】：
1. 使用 Markdown 格式。
2. 包含标题、参与人、决策结论、待办事项。
3. 去除口语废话。
4. 结构化输出，便于后续处理。
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3,
                max_tokens=settings.LOCAL_LLM_MAX_TOKENS
            )
            content = response.choices[0].message.content
            
            # 清理思考过程
            content = remove_thinking_tags(content)
            
            # 添加高亮标记
            content = add_highlighting(content)
            
            usage = response.usage
            tokens = (usage.total_tokens if usage else 0)
            logger.info(f"✅ 生成完成，消耗 Token: {tokens}")
            return content
        except Exception as e:
            logger.error(f"❌ 生成失败: {e}")
            raise LLMServiceException(f"本地LLM生成失败: {str(e)}")
    
    def _get_template(self, template_id: str) -> Dict[str, str]:
        """
        获取模板配置
        
        Args:
            template_id: 模板ID
        
        Returns:
            模板配置字典，包含 system_prompt 和 user_prompt_template
        """
        templates = {
            "default": {
                "system_prompt": "你是一个专业的高级秘书，负责将语音识别的文本整理成结构清晰的 Markdown 会议纪要。",
                "user_prompt_template": """
请根据以下内容生成会议纪要。

【参考历史信息】：
{context}

【原始语音文本】：
{raw_text}

【要求】：
1. 使用 Markdown 格式。
2. 包含标题、参与人、决策结论、待办事项。
3. 去除口语废话。
4. 结构化输出，便于后续处理。
"""
            },
        }
        
        return templates.get(template_id, templates["default"])
    
    def chat(self, prompt: str, temperature: float = 0.7, max_tokens: int = 2000) -> str:
        """
        简单的聊天接口（用于新的动态模板系统）
        
        Args:
            prompt: 完整的提示词
            temperature: 生成温度
            max_tokens: 最大token数
        
        Returns:
            模型生成的文本
        """
        logger.info("💬 本地LLM Chat 调用...")
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            
            # 清理思考过程
            content = remove_thinking_tags(content)
            
            # 添加高亮标记
            content = add_highlighting(content)
            
            logger.info(f"✅ 本地LLM 生成完成，长度: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"❌ 本地LLM Chat 调用失败: {e}")
            raise
    
    def _extract_json_from_text(self, text: str) -> dict:
        """
        从文本中提取JSON（兼容不支持json_object模式的模型）
        
        Args:
            text: 包含JSON的文本
        
        Returns:
            解析后的字典
        """
        try:
            # 尝试直接解析
            return json.loads(text)
        except json.JSONDecodeError:
            # 尝试查找JSON代码块
            import re
            json_match = re.search(r'```json\s*(\{.*?\})\s*```', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            
            # 尝试查找裸JSON
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
            
            # 解析失败，返回默认值
            logger.warning(f"⚠️ 无法从文本中提取JSON: {text[:200]}")
            return {"need_rag": False, "search_query": ""}


# 创建单例实例（延迟初始化）
_local_llm_service_instance = None

def get_local_llm_service():
    """获取本地LLM服务单例"""
    global _local_llm_service_instance
    if _local_llm_service_instance is None:
        _local_llm_service_instance = LocalLLMService()
    return _local_llm_service_instance
