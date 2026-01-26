# app/services/highlighter.py
from LAC import LAC
import re

class TextHighlighter:
    def __init__(self):
        # 1. 加载轻量级 NLP 模型 (CPU 运行即可，极快)
        print("📝 正在加载 NLP 模型 (LAC)...")
        self.lac = LAC(mode='lac')
        
        # 2. 定义【自定义项目库】
        # 这里填入你们公司的项目名、产品名、专有名词
        self.custom_projects = [
            "智融数仓", "智能提效", "辅助办公平台", 
            "会议纪要", "知识图谱", "AI写作", 
            "国政通", "Ubuntu", "Docker"
        ]

    def process(self, text: str):
        """
        输入纯文本，输出带 HTML 高亮标签的文本
        """
        if not text:
            return text

        # === 阶段 1: 通用实体识别 (人名、时间) ===
        # result 格式: [['我', '是', '张三', '今天'], ['r', 'v', 'PER', 'TIME']]
        _inputs = self.lac.run(text)
        words = _inputs[0]
        tags = _inputs[1]
        
        processed_tokens = []
        
        for word, tag in zip(words, tags):
            # 处理人名 (PER)
            if tag == 'PER':
                processed_tokens.append(f'<span class="highlight-person">{word}</span>')
            # 处理时间 (TIME)
            elif tag == 'TIME':
                processed_tokens.append(f'<span class="highlight-time">{word}</span>')
            # 处理地点 (LOC)
            elif tag == 'LOC':
                processed_tokens.append(f'<span class="highlight-loc">{word}</span>')
            else:
                processed_tokens.append(word)
        
        # 重新组合成字符串
        html_text = "".join(processed_tokens)
        
        # === 阶段 2: 自定义项目名高亮 (正则替换) ===
        # 使用正则进行精准替换，忽略大小写
        for project in self.custom_projects:
            # 这里的逻辑是：把 "项目名" 替换为 "<span class='highlight-project'>项目名</span>"
            # 为了防止重复替换 HTML 标签里的内容，这里简单处理，实际可用更复杂的正则
            pattern = re.compile(re.escape(project), re.IGNORECASE)
            html_text = pattern.sub(
                f'<span class="highlight-project">{project}</span>', 
                html_text
            )
            
        return html_text

    def check_doubtful_content(self, text: str, llm_service=None):
        """
        【进阶功能】使用 LLM 标记存疑内容
        如果传入了 LLM 服务实例，就调用它来纠错
        """
        if not llm_service:
            return text
            
        # 构造 Prompt 让大模型帮你找茬
        prompt = f"""
        你是一个专业的会议记录校对员。
        请阅读以下语音识别生成的文本。如果发现有**逻辑不通、明显识别错误、或语句不连贯**的地方，
        请用 <span class="highlight-doubt">...</span> 将其包裹起来。
        其他人名、项目名不要动。直接输出处理后的 HTML 文本，不要解释。

        文本内容：
        {text}
        """
        # 这里假设你有一个 call_llm 的方法
        try:
            return llm_service.chat(prompt)
        except:
            return text

# 单例模式
highlighter = TextHighlighter()