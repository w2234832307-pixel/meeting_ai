"""
配置管理模块
包含配置验证、环境检查等功能
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv
from app.core.exceptions import ConfigurationException

# 延迟导入logger，避免循环导入
def get_logger():
    from app.core.logger import logger
    return logger

# 1. 自动加载 .env 文件（支持多种编码）
env_path = Path(__file__).parent.parent.parent / ".env"
if env_path.exists():
    try:
        # 优先尝试UTF-8编码
        load_dotenv(dotenv_path=env_path, encoding='utf-8')
    except UnicodeDecodeError:
        # 如果UTF-8失败，尝试GBK编码（Windows中文环境）
        try:
            load_dotenv(dotenv_path=env_path, encoding='gbk')
        except Exception as e:
            # 如果都失败，尝试系统默认编码
            load_dotenv(dotenv_path=env_path, encoding='latin-1')


class Settings:
    """应用配置类 - 支持配置验证"""
    
    # --- 基础配置 ---
    PROJECT_NAME: str = "Meeting AI Service"
    VERSION: str = "1.0.0"
    API_V1_STR: str = "/api/v1"
    
    # 模式: API / LOCAL
    AI_MODE: str = os.getenv("AI_MODE", "API")
    
    # --- 数据库 (MySQL) ---
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "localhost")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "")
    MYSQL_DB: str = os.getenv("MYSQL_DB", "meeting_db")

    # --- 向量数据库配置（Chroma）---
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chroma")
    CHROMA_HOST: str = os.getenv("CHROMA_HOST", "192.168.211.74")
    CHROMA_PORT: int = int(os.getenv("CHROMA_PORT", "8000"))
    CHROMA_COLLECTION_NAME: str = os.getenv("CHROMA_COLLECTION_NAME", "employee_voice_library")
    
    # --- ASR服务配置 ---
    ASR_SERVICE_TYPE: str = os.getenv("ASR_SERVICE_TYPE", "tencent")  # tencent / funasr
    
    # 腾讯云 ASR 配置
    TENCENT_SECRET_ID: str = os.getenv("TENCENT_SECRET_ID", "")
    TENCENT_SECRET_KEY: str = os.getenv("TENCENT_SECRET_KEY", "")
    TENCENT_REGION: str = os.getenv("TENCENT_REGION", "ap-beijing")
    TENCENT_ASR_ENGINE_MODEL_TYPE: str = os.getenv("TENCENT_ASR_ENGINE_MODEL_TYPE", "16k_zh")
    
    # FunASR 配置（支持独立服务或本地模式）
    FUNASR_SERVICE_URL: str = os.getenv("FUNASR_SERVICE_URL", "")  # 如：http://localhost:8002
    FUNASR_MODEL_NAME: str = os.getenv("FUNASR_MODEL_NAME", "paraformer-zh")
    FUNASR_MODEL_REVISION: str = os.getenv("FUNASR_MODEL_REVISION", "v2.0.4")
    FUNASR_DEVICE: str = os.getenv("FUNASR_DEVICE", "cpu")  # cpu / cuda:0
    FUNASR_NCPU: int = int(os.getenv("FUNASR_NCPU", "4"))
    FUNASR_BATCH_SIZE: int = int(os.getenv("FUNASR_BATCH_SIZE", "300"))
    
    # --- Embedding服务配置 ---
    EMBEDDING_SERVICE: str = os.getenv("EMBEDDING_SERVICE", "bge-m3")  # bge-m3 / tencent / openai
    
    # 腾讯云 NLP/Embedding 配置（已弃用，保留兼容性）
    TENCENT_NLP_SECRET_ID: str = os.getenv("TENCENT_NLP_SECRET_ID", "")
    TENCENT_NLP_SECRET_KEY: str = os.getenv("TENCENT_NLP_SECRET_KEY", "")
    
    # BGE-M3 本地配置（推荐）
    BGE_M3_MODEL_NAME: str = os.getenv("BGE_M3_MODEL_NAME", "BAAI/bge-m3")
    BGE_M3_DEVICE: str = os.getenv("BGE_M3_DEVICE", "cpu")  # cpu / cuda
    BGE_M3_USE_FP16: bool = os.getenv("BGE_M3_USE_FP16", "True").lower() == "true"
    BGE_M3_BATCH_SIZE: int = int(os.getenv("BGE_M3_BATCH_SIZE", "12"))
    BGE_M3_MAX_LENGTH: int = int(os.getenv("BGE_M3_MAX_LENGTH", "8192"))
    
    # --- LLM服务配置 ---
    LLM_SERVICE_TYPE: str = os.getenv("LLM_SERVICE_TYPE", "api")  # api / local
    
    # API LLM 配置 (DeepSeek/OpenAI)
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
    
    # 本地 LLM 配置 (Qwen3-14b等)
    LOCAL_LLM_BASE_URL: str = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1")
    LOCAL_LLM_API_KEY: str = os.getenv("LOCAL_LLM_API_KEY", "")
    LOCAL_LLM_MODEL_NAME: str = os.getenv("LOCAL_LLM_MODEL_NAME", "qwen3-14b")
    LOCAL_LLM_MAX_TOKENS: int = int(os.getenv("LOCAL_LLM_MAX_TOKENS", "4096"))
    LOCAL_LLM_SUPPORT_JSON_MODE: bool = os.getenv("LOCAL_LLM_SUPPORT_JSON_MODE", "False").lower() == "true"
    LOCAL_LLM_TEST_ON_INIT: bool = os.getenv("LOCAL_LLM_TEST_ON_INIT", "True").lower() == "true"
    
    # --- 应用服务配置 ---
    APP_HOST: str = os.getenv("APP_HOST", "0.0.0.0")
    APP_PORT: int = int(os.getenv("APP_PORT", "8001"))
    
    # --- 服务超时配置 ---
    # ASR超时：5小时音频 ≈ 需要30-60分钟处理，设置为2小时（7200秒）防止超时
    ASR_TIMEOUT: int = int(os.getenv("ASR_TIMEOUT", "7200"))  # 2小时（适应最大5小时音频）
    LLM_TIMEOUT: int = int(os.getenv("LLM_TIMEOUT", "180"))  # 3分钟（大文本生成需要更长时间）
    EMBEDDING_TIMEOUT: int = int(os.getenv("EMBEDDING_TIMEOUT", "60"))  # 1分钟
    
    # --- 重试配置 ---
    ASR_MAX_RETRIES: int = int(os.getenv("ASR_MAX_RETRIES", "3"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "3"))
    EMBEDDING_MAX_RETRIES: int = int(os.getenv("EMBEDDING_MAX_RETRIES", "3"))
    
    # --- 文件上传限制 ---
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "500"))
    MAX_TEXT_LENGTH: int = int(os.getenv("MAX_TEXT_LENGTH", "100000"))

    # --- 路径配置 ---
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    LOG_DIR: Path = BASE_DIR / "logs"
    TEMP_DIR: Path = BASE_DIR / "temp_files"  # 临时存音频文件
    
    # --- 音频时长限制（秒）- 5小时 = 18000秒 ---
    MAX_AUDIO_DURATION_SECONDS: int = int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "18000"))
    
    def validate(self) -> None:
        """
        验证必需配置项
        
        Raises:
            ConfigurationException: 配置验证失败时抛出
        """
        errors = []
        
        # 验证ASR配置
        if self.ASR_SERVICE_TYPE == "tencent":
            if not self.TENCENT_SECRET_ID:
                errors.append("TENCENT_SECRET_ID 未配置（ASR_SERVICE_TYPE=tencent 时需要）")
            if not self.TENCENT_SECRET_KEY:
                errors.append("TENCENT_SECRET_KEY 未配置（ASR_SERVICE_TYPE=tencent 时需要）")
        elif self.ASR_SERVICE_TYPE == "funasr":
            # FunASR本地部署，不需要验证密钥
            pass
        else:
            errors.append(f"不支持的ASR_SERVICE_TYPE: {self.ASR_SERVICE_TYPE}")
        
        # 验证LLM配置
        if self.LLM_SERVICE_TYPE == "api":
            if not self.LLM_API_KEY:
                errors.append("LLM_API_KEY 未配置（LLM_SERVICE_TYPE=api 时需要）")
        elif self.LLM_SERVICE_TYPE == "local":
            if not self.LOCAL_LLM_BASE_URL:
                errors.append("LOCAL_LLM_BASE_URL 未配置（LLM_SERVICE_TYPE=local 时需要）")
        else:
            errors.append(f"不支持的LLM_SERVICE_TYPE: {self.LLM_SERVICE_TYPE}")
        
        # 验证Chroma配置（如果使用向量检索）
        if self.VECTOR_STORE_TYPE == "chroma":
            if not self.CHROMA_HOST:
                errors.append("CHROMA_HOST 未配置")
            if not self.CHROMA_COLLECTION_NAME:
                errors.append("CHROMA_COLLECTION_NAME 未配置")
        
        if errors:
            error_msg = "配置验证失败:\n" + "\n".join(f"  - {e}" for e in errors)
            get_logger().error(error_msg)
            raise ConfigurationException(error_msg)
        
        get_logger().info("✅ 配置验证通过")
    
    def is_tencent_nlp_available(self) -> bool:
        """检查腾讯云NLP服务是否可用"""
        return bool(self.TENCENT_NLP_SECRET_ID and self.TENCENT_NLP_SECRET_KEY)
    
    def is_tencent_asr_available(self) -> bool:
        """检查腾讯云ASR服务是否可用"""
        return bool(self.TENCENT_SECRET_ID and self.TENCENT_SECRET_KEY)


# 实例化单例
settings = Settings()

# 自动创建必要的文件夹
settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
settings.TEMP_DIR.mkdir(parents=True, exist_ok=True)

# 注意：这里不自动验证，避免在导入时就失败
# 应用启动时应该调用 settings.validate()
