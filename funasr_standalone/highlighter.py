import re
import jieba
import jieba.posseg as pseg  # 引入词性标注

class TextHighlighter:
    def __init__(self):
        print("📝 正在加载 Jieba 高亮模型...")
        # 预热一下 jieba (首次运行会加载词典)
        jieba.initialize()
        
        # 定义需要高亮的项目关键词 (支持正则)
        self.custom_projects = [
            "智融数仓", "智能提效", "辅助办公平台", 
            "会议纪要", "知识图谱", "AI写作", "国政通", 
            "FunASR", "Paraformer", "Ubuntu", "Docker"
        ]
        
        # 将自定义词加入 jieba 词典，防止被切碎
        for proj in self.custom_projects:
            jieba.add_word(proj)

    def process(self, text: str):
        """输入纯文本，输出带 HTML 高亮标签的文本"""
        if not text:
            return text

        # 1. 使用 jieba 进行分词和词性标注
        # words 格式: pair('张三', 'nr'), pair('今天', 't')
        words = pseg.cut(text)
        
        processed_tokens = []
        
        for word, flag in words:
            # nr = 人名
            if flag.startswith('nr'): 
                processed_tokens.append(f'<span style="color:#d9534f;font-weight:bold;">{word}</span>')
            # t = 时间
            elif flag.startswith('t'): 
                processed_tokens.append(f'<span style="color:#27ae60;font-weight:bold;">{word}</span>')
            # ns = 地点
            elif flag.startswith('ns'): 
                processed_tokens.append(f'<span style="color:#5bc0de;font-weight:bold;">{word}</span>')
            else:
                processed_tokens.append(word)
        
        html_text = "".join(processed_tokens)
        
        # 2. 自定义项目名高亮 (正则补漏)
        # 虽然上面 add_word 了，但为了颜色样式统一，还是扫一遍正则
        for project in self.custom_projects:
            pattern = re.compile(re.escape(project), re.IGNORECASE)
            html_text = pattern.sub(
                f'<span style="color:#2980b9;text-decoration:underline;font-weight:bold;">{project}</span>', 
                html_text
            )
            
        return html_text

# 全局初始化
highlighter = TextHighlighter()