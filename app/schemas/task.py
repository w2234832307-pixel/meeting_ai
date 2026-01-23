from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# 逐字稿单项数据模型
class TranscriptItem(BaseModel):
    """逐字稿单项"""
    text: str = Field(..., description="文本内容")
    start_time: float = Field(..., description="开始时间（秒）")
    end_time: float = Field(..., description="结束时间（秒）")
    speaker_id: Optional[int] = Field(None, description="说话人ID（如果启用说话人分离）")

# 这是返给 Java 的最终结果格式
class MeetingResponse(BaseModel):
    """会议处理响应"""
    # 状态: success / failed / error
    status: str = Field(..., description="任务状态")
    
    # 错误信息 (如果有)
    message: str = Field("", description="提示信息")
    
    # ASR 识别出的原始文本（合并后的完整文本）
    raw_text: str = Field("", description="语音转写的原始文本")
    
    # 带时间戳和说话人的逐字稿
    transcript: List[TranscriptItem] = Field(default_factory=list, description="带时间戳和说话人的逐字稿")
    
    # LLM 判断是否用了 RAG
    need_rag: bool = Field(False, description="是否触发了历史检索")
    
    # 最终生成的结构化数据（根据模板生成）
    html_content: Optional[str] = Field(None, description="HTML格式的纪要")
    
    # 消耗的 Token (方便你统计成本)
    usage_tokens: int = Field(0, description="LLM 消耗的 token 数")

class ArchiveRequest(BaseModel):
    minutes_id: int = Field(..., description="MySQL里的会议纪要ID (minutes_draft_id)")
    markdown_content: str = Field(..., description="用户修改确认后的最终版 Markdown 内容")
    user_id: Optional[str] = Field(None, description="操作人的ID (可选)")

class ArchiveResponse(BaseModel):
    status: str
    message: str
    chunks_count: int = Field(0, description="切分成了多少个片段存入向量库")