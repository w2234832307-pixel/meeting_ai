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
    # 这里不再删除 '<' 和 '>'，避免把正常的 <p ...> 变成 p ...
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
    增强版高亮函数
    """
    if not text:
        return text

    # 如果内容本身就是合法 JSON（例如说话人摘要的结构化输出），为了避免破坏 JSON 结构，直接跳过高亮
    stripped = text.strip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            json.loads(stripped)
            return text
        except Exception:
            # 不是纯 JSON，再继续后面的高亮逻辑
            pass

    # 定义样式
    STYLES = {
        "person": 'style="background-color: #dbeafe; color: #1e40af; padding: 0 2px; border-radius: 3px;"',
        "date": 'style="background-color: #dcfce7; color: #166534; padding: 0 2px; border-radius: 3px;"',
        "uncertain": 'style="background-color: #fee2e2; color: #991b1b; text-decoration: underline dashed;"',
        "project": 'style="background-color: #f3e8ff; color: #6b21a8; font-weight: 500;"'
    }

    # === 1. 人名优化 ===
    # 1.1 匹配 Markdown 加粗/Strong/引号 (保留原逻辑)
    text = re.sub(r'["“]([一-龥]{1,4})["”]', f'<mark {STYLES["person"]}>\\1</mark>', text) # 增加了中文引号支持
    text = re.sub(r'<strong>([一-龥]{1,10})</strong>', f'<mark {STYLES["person"]}>\\1</mark>', text)
    text = re.sub(r'\*\*([一-龥]{1,10})\*\*', f'<mark {STYLES["person"]}>\\1</mark>', text)
    
    # 1.2 [新增] 匹配 "姓+称谓" (简单版NER)
    # 避免匹配到 "总共" 里的 "总"，要求前面是人名常见的字
    text = re.sub(
        r'([张王李赵刘陈杨黄吴周徐孙马朱胡林郭何高罗][一-龥]{0,2})(经理|总|老师|工|董|总监|组长)',
        f'<mark {STYLES["person"]}>\\1\\2</mark>',
        text
    )

    # === 2. 项目名优化 ===
    project_patterns = [
        # 2.1 英文大写 (排除常用非项目词)
        (r'\b(?!(?:ID|OK|NO|Yes|HI|BYE|TODO|PPT|PDF|WORD|EXCEL|CEO|CTO|CFO|HR|KPI)\b)([A-Z]{2,10})\b', 
         f'<mark {STYLES["project"]}>\\1</mark>'),
        
        # 2.2 中文项目名 (收紧匹配范围，排除 "的" "了" "是" 等开头)
        # {2,12} 限制长度，[^...] 排除常用虚词开头
        (r'(?<![一-龥])([a-zA-Z0-9\u4e00-\u9fa5]{2,12}(?:项目|产品|系统|平台|工具|服务|计划|方案|中台|大脑))', 
         f'<mark {STYLES["project"]}>\\1</mark>'),
    ]
    for pattern, replacement in project_patterns:
        text = re.sub(pattern, replacement, text)

    # === 3. 日期 (保留原逻辑，效果已经不错) ===
    date_patterns = [
        (r'(周[一二三四五六日天])', f'<mark {STYLES["date"]}>\\1</mark>'),
        (r'(今天|明天|后天|昨天|前天)', f'<mark {STYLES["date"]}>\\1</mark>'),
        (r'(本周|下周|上周|这周|上上周)', f'<mark {STYLES["date"]}>\\1</mark>'),
        (r'(本月|下月|上月|这个月)', f'<mark {STYLES["date"]}>\\1</mark>'),
        (r'(\d{1,2}月\d{1,2}日)', f'<mark {STYLES["date"]}>\\1</mark>'),
        (r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})', f'<mark {STYLES["date"]}>\\1</mark>'),
        (r'(\d{1,2}:\d{2})', f'<mark {STYLES["date"]}>\\1</mark>'), # 新增：支持 14:00 这种时间
    ]
    for pattern, replacement in date_patterns:
        text = re.sub(pattern, replacement, text)

    # === 4. 存疑 (保留原逻辑) ===
    uncertain_patterns = [
        (r'(?:【|\[)存疑[：:]\s*([^】\]]+)(?:】|\])', f'<mark {STYLES["uncertain"]}>\\1</mark>'),
    ]
    for pattern, replacement in uncertain_patterns:
        text = re.sub(pattern, replacement, text)

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
        # 最近一次调用的 token 使用情况（用于上层统计）
        self.last_usage: Optional[Dict[str, int]] = None
        
        # 检查 API Key 是否配置
        if not self.api_key or self.api_key.strip() == "":
            logger.warning("⚠️ LLM_API_KEY 未配置！请在 .env 文件或 docker-compose.yml 中设置")
            logger.warning("   示例：LLM_API_KEY=sk-xxxxxxxxxxxxx")
        
        # 初始化 OpenAI 客户端 (兼容 DeepSeek)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
        
        logger.info(f"🕵️‍♂️ LLM 连接地址: {self.base_url}")
        logger.info(f"🤖 使用模型: {self.model}")
        if self.api_key and self.api_key.strip() != "":
            # 只显示前4位和后4位，中间用*代替
            masked_key = self.api_key[:4] + "*" * (len(self.api_key) - 8) + self.api_key[-4:] if len(self.api_key) > 8 else "***"
            logger.info(f"🔑 API Key: {masked_key}")
        else:
            logger.warning("⚠️ API Key 为空，LLM 调用将失败")

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
{user_requirement_section}

【历史背景资料 (RAG)】：
{context if context else "无"}

----------------
【会议录音转录文本】：
{raw_text}

----------------
👇👇👇 最重要的指令在下面 👇👇👇

请仔细阅读上方的【会议录音转录文本】。
现在，请你严格使用上面的录音内容，来填充下方这个【会议纪要模板】。
⚠️ 再次警告：模板中的“(待补充)”等字眼必须被替换为真实内容，如果录音里没提到，请写“会议未提及”，绝不能保留“待补充”！

【会议纪要模板结构】(请保持原样标题，只填充内容)：
{template_id}

请直接输出填充后的 Markdown 结果：
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
            
            # 添加高亮标记
            content = add_highlighting(content)
            
            usage = getattr(response, "usage", None)
            if usage:
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.last_usage = {
                    "total_tokens": int(total_tokens),
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                }
                logger.info(
                    f"✅ 生成完成，Token 使用：输入={prompt_tokens}，输出={completion_tokens}，总计={total_tokens}"
                )
            else:
                self.last_usage = None
                logger.info("✅ 生成完成，但未返回 Token 使用信息")
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
    
    def chat(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        完善后的聊天接口：强制注入系统指令，修复参数覆盖问题
        """
        logger.info("💬 LLM Chat 调用...")
        
        # 1. 强制注入强大的 System Prompt（系统护栏）
        system_instruction = """
你是一名拥有10年经验的高级会议秘书。你的任务是根据【会议录音转录文本】来填充【会议纪要模板】。
⚠️ 核心操作指南：
1. 严禁复读占位符：绝对不允许在输出中出现“（待补充）”、“（此处需补充内容）”等原模板的提示语。
2. 智能信息对齐：
   - 对于会议时间、地点、人员，如果录音中没有明确提到，请结合语境简单推断，或填写“录音未明确”。
   - 重点：如果模板要求提炼某人（如“陈总”、“张总”）的发言，而文本中只有“SPEAKER_00”等代号，请你根据上下文语境，智能寻找最匹配的说话人，并详实提炼其核心观点！不要死板地填“未提及”！
3. 内容必须丰满：对于各部门的工作通报、决策和后续跟进，请把具体的痛点、数据和方案详细写出来，尽可能详尽，绝不能一笔带过！
4. 严格输出 Markdown 格式，保持原模板的标题层级不变。
"""

        # 2. 优先使用实例属性（如果 endpoints.py 传了值就用传的，没传就用默认的 0.3）
        actual_temp = getattr(self, 'temperature', temperature)
        actual_tokens = getattr(self, 'max_tokens', max_tokens)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_instruction}, # 👈 救命的系统提示词
                    {"role": "user", "content": prompt}
                ],
                temperature=actual_temp,
                max_tokens=actual_tokens
            )
            
            content = response.choices[0].message.content
            
            # 清理思考过程 (针对 DeepSeek-R1 等带有 <think> 的模型)
            content = remove_thinking_tags(content)
            # 添加高亮标记
            content = add_highlighting(content)

            usage = getattr(response, "usage", None)
            if usage:
                total_tokens = getattr(usage, "total_tokens", 0) or 0
                prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
                completion_tokens = getattr(usage, "completion_tokens", 0) or 0
                self.last_usage = {
                    "total_tokens": int(total_tokens),
                    "prompt_tokens": int(prompt_tokens),
                    "completion_tokens": int(completion_tokens),
                }
                logger.info(
                    f"✅ LLM 生成完成，长度: {len(content)}，Token 使用：输入={prompt_tokens}，输出={completion_tokens}，总计={total_tokens}，温度={actual_temp}"
                )
            else:
                self.last_usage = None
                logger.info(f"✅ LLM 生成完成，长度: {len(content)}（未返回 Token 使用信息）")

            return content
            
        except Exception as e:
            error_msg = str(e)
            # 提供更友好的错误提示
            if "Authentication" in error_msg or "governor" in error_msg:
                if not self.api_key or self.api_key == "":
                    logger.error("❌ LLM API Key 未配置！请在 .env 文件或 docker-compose.yml 中设置 LLM_API_KEY")
                    raise ValueError("LLM API Key 未配置。请检查环境变量 LLM_API_KEY 是否正确传递到容器内。")
                else:
                    logger.error(f"❌ LLM 认证失败: {error_msg}")
                    logger.error("💡 可能的原因：")
                    logger.error("   1. API Key 已过期或无效")
                    logger.error("   2. API Key 格式错误（应包含 'sk-' 前缀）")
                    logger.error("   3. 账号被限制或禁用")
                    logger.error("   4. 余额不足或达到调用限制")
                    raise ValueError(f"LLM 认证失败: {error_msg}。请检查 API Key 是否正确且有效。")
            elif "rate limit" in error_msg.lower() or "429" in error_msg:
                logger.error(f"❌ LLM 调用频率限制: {error_msg}")
                logger.error("💡 建议：降低请求频率或检查账号配额")
                raise
            else:
                logger.error(f"❌ LLM Chat 调用失败: {error_msg}")
                raise

llm_service = LLMService()