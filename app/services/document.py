"""
文档解析服务 - 支持Word和PDF文件
"""
import os
from pathlib import Path
from typing import Optional
from app.core.logger import logger


class DocumentService:
    """文档解析服务类"""
    
    def __init__(self):
        """初始化文档服务"""
        pass
    
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


# 创建单例实例
document_service = DocumentService()