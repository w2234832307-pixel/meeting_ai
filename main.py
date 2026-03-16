"""
FastAPI 应用入口文件
"""
import os
import sys
import time
import glob
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logger import logger
from app.api.endpoints import router

# ✅ 确保标准输出使用UTF-8编码（Windows兼容）
if sys.platform == "win32":
    import io
    if hasattr(sys.stdout, 'buffer'):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    if hasattr(sys.stderr, 'buffer'):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')


# 👇 1. 定义全局清理函数
def global_temp_sweep(temp_dir: str, max_age_hours: int = 2):
    """全局兜底清扫：清理存活时间超过指定小时数的废弃文件"""
    try:
        now = time.time()
        search_pattern = os.path.join(temp_dir, "*")
        cleaned_count = 0
        for file_path in glob.glob(search_pattern):
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                if (now - file_mtime) > (max_age_hours * 3600):
                    try:
                        os.remove(file_path)
                        cleaned_count += 1
                    except:
                        pass
        if cleaned_count > 0:
            logger.info(f"♻️ 全局垃圾回收完成，清理了 {cleaned_count} 个滞留临时文件")
    except Exception as e:
        logger.error(f"❌ 全局临时文件清扫失败: {e}")


# 👇 2. 定义生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 启动时执行：打扫上一次崩溃可能遗留的战场
    logger.info("🧹 执行启动前的磁盘清理...")
    # 确保 TEMP_DIR 存在
    os.makedirs(settings.TEMP_DIR, exist_ok=True)
    global_temp_sweep(str(settings.TEMP_DIR), max_age_hours=2)
    yield
    # 关闭服务时执行的内容写在这里


# 👇 3. 唯一一次创建 FastAPI 应用实例，并挂载 lifespan
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="会议AI服务 - 支持语音转文字、智能总结、RAG检索",
    lifespan=lifespan
)

# 👇 4. 挂载中间件和路由
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router, prefix="/api/v1", tags=["会议处理"])


# 在启动时初始化 Pyannote 服务（如果配置了）
try:
    from app.services.pyannote_service import get_pyannote_service
    pyannote_service = get_pyannote_service()
    if pyannote_service.is_available():
        logger.info(f"✅ Pyannote 服务已就绪: {pyannote_service.base_url}")
    else:
        logger.info("ℹ️ 未配置 PYANNOTE_SERVICE_URL，将使用 FunASR 内置说话人分离")
except Exception as e:
    logger.warning(f"⚠️ Pyannote 服务初始化失败: {e}")


@app.get("/")
async def root():
    """根路径，健康检查"""
    return {
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "status": "running"
    }


@app.get("/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "mode": settings.AI_MODE
    }


if __name__ == "__main__":
    # 从环境变量读取端口配置
    port = int(os.getenv("APP_PORT", "8000"))
    host = os.getenv("APP_HOST", "0.0.0.0")
    
    # 打印启动信息（使用print确保显示）
    print("\n" + "="*80)
    print(f"🚀 会议AI服务启动中...")
    print(f"📋 当前模式: {settings.AI_MODE}")
    print(f"📁 日志路径: {settings.LOG_DIR}")
    print(f"🔌 监听地址: http://{host}:{port}")
    print(f"📚 API文档: http://localhost:{port}/docs")
    print("="*80 + "\n")
    
    logger.info(f"🚀 服务启动成功! 当前模式: {settings.AI_MODE}")
    logger.info(f"📁 日志路径: {settings.LOG_DIR}")
    logger.info(f"🔌 监听端口: {port}")
    
    reload_mode = os.getenv("RELOAD", "false").lower() == "true"
    
    if reload_mode:
        logger.warning("⚠️ Reload模式已启用，API调用日志可能不显示在主终端")
    
    uvicorn_config = {
        "app": "main:app",
        "host": host,
        "port": port,
        "reload": reload_mode,  # 通过环境变量控制
        "log_level": "info",
        "access_log": True,  # 显示访问日志
        "use_colors": True,  # 使用彩色输出
    }
    
    uvicorn.run(**uvicorn_config)