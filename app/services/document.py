"""
文档解析服务 - 支持Word和PDF文件
"""
import os
from pathlib import Path
from typing import Optional
from app.core.logger import logger
from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH


class DocumentService:
    """文档解析服务类
    
    说明：
    - extract_text_* 系列：只提取纯文本，用于喂给 LLM 做摘要/理解
    - extract_html_* 系列：尽量保留原始格式，生成 HTML5，用于前端展示或格式保留场景
    """
    
    def __init__(self):
        """初始化文档服务"""
        pass
    
    # ============================
    # 纯文本提取接口（原有逻辑）
    # ============================
    def extract_text_from_docx(self, file_path: str) -> Optional[str]:
        """
        从Word文档（.docx）提取文本
        
        Args:
            file_path: Word文档路径
        
        Returns:
            提取的文本内容，失败返回None
        """
        try:
            from docx import Document
            
            logger.info(f"📄 开始解析Word文档: {file_path}")
            
            doc = Document(file_path)
            
            # 提取所有段落文本
            paragraphs = []
            for para in doc.paragraphs:
                text = para.text.strip()
                if text:
                    paragraphs.append(text)
            
            # 提取表格文本
            for table in doc.tables:
                for row in table.rows:
                    row_text = []
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            row_text.append(cell_text)
                    if row_text:
                        paragraphs.append(" | ".join(row_text))
            
            full_text = "\n".join(paragraphs)
            
            logger.info(f"✅ Word文档解析完成，提取文本长度: {len(full_text)}")
            return full_text
            
        except ImportError:
            logger.error("❌ python-docx 库未安装，请运行: pip install python-docx")
            return None
        except Exception as e:
            logger.error(f"❌ Word文档解析失败: {e}")
            return None
    
    def extract_text_from_pdf(self, file_path: str) -> Optional[str]:
        """
        从PDF文档提取文本
        
        Args:
            file_path: PDF文档路径
        
        Returns:
            提取的文本内容，失败返回None
        """
        try:
            import PyPDF2
            
            logger.info(f"📄 开始解析PDF文档: {file_path}")
            
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                
                # 提取所有页面文本
                pages_text = []
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text = page.extract_text()
                    if text.strip():
                        pages_text.append(text.strip())
                
                full_text = "\n".join(pages_text)
                
                logger.info(f"✅ PDF文档解析完成，共{len(pdf_reader.pages)}页，提取文本长度: {len(full_text)}")
                return full_text
                
        except ImportError:
            logger.error("❌ PyPDF2 库未安装，请运行: pip install PyPDF2")
            return None
        except Exception as e:
            logger.error(f"❌ PDF文档解析失败: {e}")
            return None
    
    def extract_text_from_file(self, file_path: str) -> Optional[str]:
        """
        自动识别文件类型并提取文本
        
        Args:
            file_path: 文件路径
        
        Returns:
            提取的文本内容，失败返回None
        """
        if not os.path.exists(file_path):
            logger.error(f"❌ 文件不存在: {file_path}")
            return None
        
        # 获取文件扩展名
        ext = os.path.splitext(file_path)[1].lower()
        
        if ext == '.docx':
            return self.extract_text_from_docx(file_path)
        elif ext == '.pdf':
            return self.extract_text_from_pdf(file_path)
        elif ext == '.txt':
            # 纯文本文件直接读取
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read()
            except UnicodeDecodeError:
                # 尝试其他编码
                try:
                    with open(file_path, 'r', encoding='gbk') as f:
                        return f.read()
                except Exception as e:
                    logger.error(f"❌ 文本文件读取失败: {e}")
                    return None
        else:
            logger.error(f"❌ 不支持的文件格式: {ext}")
            return None

    def extract_html_from_docx(self, file_path: str):
        from docx import Document
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        
        try:
            doc = Document(file_path)
            html_paragraphs = []

            for p in doc.paragraphs:
                # 1. 第一步：地毯式搜索属性 (之前的逻辑)
                alignment = p.alignment
                if alignment is None:
                    alignment = p.paragraph_format.alignment
                if alignment is None and p.style:
                    curr_style = p.style
                    while curr_style:
                        if curr_style.paragraph_format.alignment is not None:
                            alignment = curr_style.paragraph_format.alignment
                            break
                        curr_style = getattr(curr_style, 'base_style', None)

                # 2. 第二步：语义兜底 (解决读不到属性的顽固文件)
                # 逻辑：如果字号 > 20pt (大概 26px 以上)，通常就是标题，强制居中
                is_large_font = False
                for run in p.runs:
                    # 增加 run.font.size 是否存在的检查
                    if run.font and run.font.size and run.font.size.pt:
                        if run.font.size.pt > 20:
                            is_large_font = True
                            break

                # 3. 确定最终 Class
                align_class = ""
                if alignment == 1 or is_large_font: # 1 是居中，或者字号超大
                    align_class = "center"
                elif alignment == 2:
                    align_class = "right"
                
                # --- 调试：如果还是抓不到，你可以加一行打印看看 ---
                # print(f"Text: {p.text[:10]}... | Alignment Found: {alignment}")

                # --- 3. 处理红头横线逻辑 (保持不变) ---
                line_style = ""
                if "编号" in p.text or "签发人" in p.text:
                    line_style = "border-bottom: 2px solid black; padding-bottom: 5px; margin-bottom: 15px;"

                # --- 4. 处理 Runs (文字、颜色、字号) ---
                runs_html = ""
                for run in p.runs:
                    font_css = []
                    if run.font.size:
                        # Word 字号转 px
                        font_css.append(f"font-size: {run.font.size.pt * 1.3}px")
                    if run.font.color and run.font.color.rgb:
                        font_css.append(f"color: #{run.font.color.rgb}")
                    
                    style_str = "; ".join(font_css)
                    content = run.text.replace(" ", "&nbsp;") # 保留空格
                    
                    inner_html = f'<span style="{style_str}">{content}</span>'
                    if run.bold:
                        inner_html = f"<strong>{inner_html}</strong>"
                    
                    runs_html += inner_html

                # --- 5. 组装段落标签 ---
                if runs_html.strip() or p.runs:
                    # 确保 class="{align_class}" 始终存在
                    html_paragraphs.append(
                        f'<p class="{align_class}" style="{line_style}">{runs_html}</p>'
                    )

            # --- 6. 修改这里：确保 CSS 定义严谨 ---
            full_html = f"""
            <!DOCTYPE html><html><head><meta charset="utf-8">
            <style>
                body {{ font-family: "SimSun", serif; padding: 40px; max-width: 800px; margin: 0 auto; background: #fff; }}
                /* 必须设置 width: 100% 居中才能生效 */
                p {{ 
                    margin: 0.8em 0; 
                    width: 100%; 
                    min-height: 1.2em; 
                    line-height: 1.6;
                    text-align: left; /* 默认左对齐 */
                }}
                .center {{ text-align: center !important; }}
                .right {{ text-align: right !important; }}
                strong {{ font-weight: bold; }}
            </style></head>
            <body>{"".join(html_paragraphs)}</body></html>
            """
            return full_html

        except Exception as e:
            import traceback
            return f"解析异常: {str(e)}\n{traceback.format_exc()}"


    # def extract_html_from_docx(self, file_path):
    #     doc = Document(file_path)
    #     html_paragraphs = []

    #     for p in doc.paragraphs:
    #         # --- 1. 处理对齐 ---
    #         alignment = p.alignment
        
    #         # 2. 如果显式设置是 None，则去样式（Style）里寻找
    #         if alignment is None:
    #             # 向上追溯样式的对齐方式
    #             current_style = p.style
    #             while current_style is not None:
    #                 if current_style.paragraph_format.alignment is not None:
    #                     alignment = current_style.paragraph_format.alignment
    #                     break
    #                 current_style = current_style.base_style  # 继续往父类样式找
            
    #         # 3. 翻译为 HTML 类名 (1=CENTER, 2=RIGHT)
    #         align_class = ""
    #         if alignment == 1:
    #             align_class = "center"
    #         elif alignment == 2:
    #             align_class = "right"

    #         # --- 2. 处理横线 (红头文件逻辑) ---
    #         # 如果这一行包含 "编号" 且下方有横线样式
    #         extra_style = ""
    #         if "编号：" in p.text or "【20" in p.text:
    #             extra_style = " border-bottom: 2px solid black; padding-bottom: 10px; margin-bottom: 20px;"

    #         runs_html = ""
    #         for run in p.runs:
    #             text = run.text.replace(" ", "&nbsp;") # 保留空格
                
    #             # --- 3. 处理字号与颜色 ---
    #             font_style = []
    #             if run.font.size:
    #                 # Word字号转px: pt * 1.33
    #                 font_style.append(f"font-size: {run.font.size.pt * 1.33:.1f}px")
    #             if run.font.color and run.font.color.rgb:
    #                 font_style.append(f"color: #{run.font.color.rgb}")
                
    #             style_str = "; ".join(font_style)
    #             span_html = f'<span style="{style_str}">{text}</span>'
                
    #             # 处理加粗
    #             if run.bold:
    #                 span_html = f"<strong>{span_html}</strong>"
                
    #             runs_html += span_html

    #         # --- 4. 组装标签 ---
    #         tag = "p"
    #         if p.style.name.startswith('Heading'):
    #             tag = f"h{p.style.name[-1]}"
                
    #         if runs_html.strip() or "&nbsp;" in runs_html:
    #             html_paragraphs.append(
    #                 f'<{tag} class="{align_class}" style="{extra_style}">{runs_html}</{tag}>'
    #             )

    #     # 5. 最终 CSS 补全
    #     full_html = f"""
    #     <html><head><style>
    #         body {{ font-family: "SimSun", serif; padding: 50px; line-height: 1.5; background-color: #f5f5f5; }}
    #         /* 核心：确保段落作为块级元素宽度占满，否则内部居中无效 */
    #         p {{ 
    #             margin: 10px 0; 
    #             width: 100%; 
    #             min-height: 1em;
    #             white-space: pre-wrap; /* 保留空格 */
    #         }}
    #         /* 居中和右对齐强制生效 */
    #         .center {{ text-align: center !important; }}
    #         .right {{ text-align: right !important; }}
            
    #         h1, h2 {{ font-weight: bold; text-align: center; }}
    #         strong {{ font-weight: bold; }}
    #     </style></head>
    #     <body>{"".join(html_paragraphs)}</body></html>
    #     """
    #     return full_html


    def extract_html_from_pdf(self, file_path: str) -> Optional[str]:
        """
        从 PDF 文档提取带格式的 HTML。
        
        说明：
        - 高保真还原依赖外部工具 pdf2htmlEX（推荐），需要系统已安装该命令。
        - 如果 pdf2htmlEX 不可用，则退化为简单的文本包装成 <pre> 的 HTML。
        """
        import subprocess
        import tempfile

        logger.info(f"📄 开始将 PDF 文档转换为 HTML: {file_path}")

        # 先尝试使用 pdf2htmlEX（需要系统安装）
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_html:
                output_path = tmp_html.name

            cmd = [
                "pdf2htmlEX",
                "--embed", "cfijo",  # 内嵌字体/图片/JS/CSS，方便单文件传输
                "--dest-dir", os.path.dirname(output_path),
                "--output", os.path.basename(output_path),
                file_path,
            ]
            logger.info(f"🔧 调用 pdf2htmlEX: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if os.path.exists(output_path):
                with open(output_path, "r", encoding="utf-8", errors="ignore") as f:
                    html_content = f.read()
                logger.info(f"✅ PDF->HTML 转换完成（pdf2htmlEX），长度: {len(html_content)}")
                return html_content
        except FileNotFoundError:
            logger.warning("⚠️ 未找到 pdf2htmlEX 命令，将退化为简单的文本 HTML。请在系统中安装 pdf2htmlEX 以获得高保真还原。")
        except subprocess.CalledProcessError as e:
            logger.warning(f"⚠️ pdf2htmlEX 转换失败，退化为简单文本 HTML: {e}")
        except Exception as e:
            logger.warning(f"⚠️ pdf2htmlEX 调用异常，退化为简单文本 HTML: {e}")

        # 退化方案：使用纯文本 + <pre> 包裹（不会高保真，但至少可用）
        text = self.extract_text_from_pdf(file_path)
        if text is None:
            return None
        safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        fallback_html = (
            "<!DOCTYPE html>\n"
            "<html><head><meta charset=\"utf-8\"></head><body>\n"
            "<pre style=\"white-space: pre-wrap;\">\n"
            f"{safe_text}\n"
            "</pre>\n"
            "</body></html>"
        )
        logger.info(f"✅ 使用退化方案生成 PDF 文本 HTML，长度: {len(fallback_html)}")
        return fallback_html

    def extract_html_from_file(self, file_path: str) -> Optional[str]:
        """
        自动识别文件类型并提取 HTML（尽量保留原始格式）
        
        - .docx -> 语义化 HTML（mammoth）
        - .pdf  -> 优先 pdf2htmlEX，高保真 HTML；否则退化为 <pre> 文本 HTML
        - .txt  -> 简单包裹到 <pre> 中
        """
        if not os.path.exists(file_path):
            logger.error(f"❌ 文件不存在: {file_path}")
            return None

        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".docx":
            return self.extract_html_from_docx(file_path)
        elif ext == ".pdf":
            return self.extract_html_from_pdf(file_path)
        elif ext == ".txt":
            # 纯文本文件：简单用 <pre> 包装
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="gbk") as f:
                        text = f.read()
                except Exception as e:
                    logger.error(f"❌ 文本文件读取失败: {e}")
                    return None
            safe_text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            html = (
                "<!DOCTYPE html>\n"
                "<html><head><meta charset=\"utf-8\"></head><body>\n"
                "<pre style=\"white-space: pre-wrap;\">\n"
                f"{safe_text}\n"
                "</pre>\n"
                "</body></html>"
            )
            logger.info(f"✅ 文本文件转 HTML 完成，长度: {len(html)}")
            return html
        else:
            logger.error(f"❌ 不支持的文件格式(HTML提取): {ext}")
            return None


# 创建单例实例
document_service = DocumentService()