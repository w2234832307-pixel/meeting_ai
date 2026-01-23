import json
import re
import time
from typing import Dict
from openai import OpenAI, APITimeoutError, APIConnectionError
from app.core.config import settings
from app.core.logger import logger


def remove_thinking_tags(text: str) -> str:
    """
    移除LLM输出中的思考过程标签
    支持多种格式：
    1. <think>...</think>
    2. <p>...思考内容...</p>...<h3>会议纪要</h3>
    3. HTML嵌套的各种变体
    """
    if not text:
        return text
    
    original_length = len(text)
    
    # === 策略1: 移除标准 <think> 标签 ===
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # === 策略2: 移除非标准格式 - 从开头到第一个 Markdown 标题之前的所有内容 ===
    # 检测是否以 <p> 开头，且后面有 Markdown 标题（###、##、#）
    match = re.search(r'^[\s\S]*?(?=#{1,3}\s)', text)
    if match and match.group(0).strip().startswith('<p>'):
        # 移除从开头到第一个标题之前的所有 HTML 段落（思考内容）
        text = re.sub(r'^.*?(?=#{1,3}\s)', '', text, flags=re.DOTALL)
        logger.info("🧹 检测到非标准思考格式，已移除开头的 HTML 段落")
    
    # === 策略3: 移除包含思考关键词的 <p> 段落 ===
    # 常见思考关键词：好的、首先、接下来、需要注意、最后
    thinking_keywords = [
        r'<p>[\s\S]*?好的，我需要.*?</p>',
        r'<p>[\s\S]*?首先.*?</p>',
        r'<p>[\s\S]*?接下来.*?</p>',
        r'<p>[\s\S]*?需要注意.*?</p>',
        r'<p>[\s\S]*?最后，需要确保.*?</p>',
        r'<p>[\s\S]*?</think></p>',  # 残留的 </think> 标签
    ]
    for pattern in thinking_keywords:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # === 清理残留 ===
    # 移除空的 <p> 标签
    text = re.sub(r'<p>\s*</p>', '', text, flags=re.DOTALL)
    
    # 移除开头的 <p> 和引号（如果还有残留）
    text = re.sub(r'^[\s"]*<p>\s*', '', text)
    
    # 移除多余的空白行
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    
    # 去除开头和结尾的空白和引号
    text = text.strip().strip('"').strip()
    
    removed_chars = original_length - len(text)
    if removed_chars > 0:
        logger.info(f"🧹 已清理思考内容: 移除 {removed_chars} 字符")
    
    return text

class LLMService:
    def __init__(self, api_key: str = None, base_url: str = None, model_name: str = None):
        """
        初始化 LLM 服务
        
        Args:
            api_key: API密钥（如果为None，使用配置文件）
            base_url: API地址（如果为None，使用配置文件）
            model_name: 模型名称（如果为None，使用配置文件）
        """
        # 使用传入参数或配置文件值
        self.api_key = api_key or settings.LLM_API_KEY
        self.base_url = base_url or settings.LLM_BASE_URL
        self.model = model_name or settings.LLM_MODEL_NAME
        
        # 初始化 OpenAI 客户端 (兼容 DeepSeek)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"🕵️‍♂️ LLM 连接地址: {self.base_url}")
        logger.info(f"🤖 使用模型: {self.model}")

    def judge_rag(self, raw_text: str, template_id: str) -> dict:
        """
        不仅判断是否需要搜，还要生成“搜什么”
        """
        logger.info("🧠 LLM 正在分析 RAG 意图并提取关键词...")
        
        # 我们把整段文本传进去（或者截取前 2000 字，取决于 LLM 上下文窗口）
        # 让 LLM 忽略废话，提取核心实体
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

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1,
                    response_format={"type": "json_object"},
                    timeout=10  # 设置超时时间10秒
                )
                content = response.choices[0].message.content
                return json.loads(content)
                
            except (APITimeoutError, APIConnectionError) as e:
                if attempt < max_retries - 1:
                    logger.warning(f"⚠️ 网络波动，正在第 {attempt+1} 次重试 RAG 分析...")
                    time.sleep(2)  
                else:
                    logger.error(f"❌ RAG 分析最终失败: {e}")
            except Exception as e:
                # 其他错误（如代码逻辑错）不重试，直接退出
                logger.error(f"❌ RAG 分析逻辑错误: {e}")
                break
        
        # 兜底返回
        return {"need_rag": False, "search_query": ""}
        
    def generate_markdown(
        self, 
        raw_text: str, 
        context: str = "", 
        template_id: str = "default", 
        custom_instruction: str = None  # <--- 接收参数
    ) -> str:
        logger.info(f"🧠 LLM 正在生成数据... (模板指示长度: {len(template_id)})")
        
        # ------------------------------------------------------------------
        # 1. 处理用户指令 (User Instruction)
        # ------------------------------------------------------------------
        user_requirement_section = ""
        if custom_instruction and custom_instruction.strip():
            user_requirement_section = f"""
### 🔥 用户特别强调的要求 (最高优先级)
用户对本次生成有以下具体指示，请**务必满足**：
> "{custom_instruction}"
"""

        # ------------------------------------------------------------------
        # 2. 定义核心指令 (System Prompt)
        # ------------------------------------------------------------------
        core_instruction = """
请严格遵守以下填空规则：
你是一名拥有10年经验的高级会议秘书。你的任务是根据【会议录音转录文本】，精确填充用户提供的【会议纪要模板】。

规则：
1. **格式强制**：必须使用 Markdown 标准标题语法 (#, ##, ###)，严禁仅使用加粗。
2. **智能填空**：根据录音内容提取时间、人员、决议。
3. **内容映射**：若录音中无对应内容，填"无"或留空，不可编造。
4. **语气风格**：客观、简练、专业。
5. **输出要求**：直接输出会议纪要内容，不要包含思考过程、不要使用<think>标签、不要输出任何额外的HTML标签。
"""

        # ------------------------------------------------------------------
        # 3. 动态构建 Prompt
        # ------------------------------------------------------------------
        # 判断是文件内容还是ID
        is_custom_content = len(template_id) > 50 and ("\n" in template_id or "\r" in template_id)
        
        system_prompt = core_instruction
        
        if is_custom_content:
            # === 情况 A: 用户提供了 Word 里的具体内容 ===
            logger.info("📄 识别到自定义模板内容，使用动态提示词构建...")
            
            user_input = f"""
请根据以下录音文本，严格按照【会议纪要模板】的格式生成内容。

{user_requirement_section}

----------------
【会议纪要模板结构】(请完全照搬此结构填充)：
{template_id}

----------------
【历史背景资料 (RAG)】：
{context if context else "无"}

----------------
【会议录音转录文本】：
{raw_text}

----------------
请开始生成：
"""
        else:
            # === 情况 B: 传入的是 default 这种简短 ID ===
            logger.info(f"🔑 使用预设模板 ID: {template_id}")
            template_config = self._get_template(template_id)
            
            if "system_prompt" in template_config:
                system_prompt = template_config["system_prompt"]
            
            user_input = template_config.get("user_prompt_template", "").format(
                context=context if context else "无",
                raw_text=raw_text,
                user_requirement_section=user_requirement_section 
            )

        # ------------------------------------------------------------------
        # 4. 调用 LLM
        # ------------------------------------------------------------------
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_input}
                ],
                temperature=0.3
            )
            content = response.choices[0].message.content
            
            # 清理思考过程
            content = remove_thinking_tags(content)
            
            usage = response.usage
            tokens = (usage.total_tokens if usage else 0)
            logger.info(f"✅ 生成完成，消耗 Token: {tokens}")
            return content
        except Exception as e:
            logger.error(f"❌ 生成失败: {e}")
            return f"生成失败，错误信息: {str(e)}"

    def _get_template(self, template_id: str) -> Dict[str, str]:
        templates = {
            "default": {
                "system_prompt": "你是一个专业的高级秘书，负责将语音识别的文本整理成结构清晰的 Markdown 会议纪要。",
                
                "user_prompt_template": """
请根据以下内容生成会议纪要。

【参考历史信息】：
{context}

【Task】：
根据下方的【Meeting Transcript】，生成一份专业的会议纪要。

{user_requirement_section}

【原始语音文本】：
{raw_text}

【要求】：
1. 使用 Markdown 格式。
2. 包含标题、参与人、决策结论、待办事项。
3. 去除口语废话。
"""
            }
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
        logger.info("💬 LLM Chat 调用...")
        
        try:
            print("-" * 30)
            print(f"🕵️ [Debug] 正在请求的 API 地址 (Base URL): {self.client.base_url}") 
            print(f"🕵️ [Debug] 使用的模型名称: {self.model}")                           
            key_preview = str(self.client.api_key)[:8] if self.client.api_key else "None"
            print(f"🕵️ [Debug] 使用的 API Key: {key_preview}...")                      
            print("-" * 30)
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            content = response.choices[0].message.content
            # 清理思考过程
            content = remove_thinking_tags(content)
            logger.info(f"✅ LLM 生成完成，长度: {len(content)}")
            return content
            
        except Exception as e:
            logger.error(f"❌ LLM Chat 调用失败: {e}")
            raise

llm_service = LLMService()