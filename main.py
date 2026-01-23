"""
FastAPI 应用入口文件
"""
import os
import sys
import uvicorn
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

# 创建 FastAPI 应用实例
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="会议AI服务 - 支持语音转文字、智能总结、RAG检索"
)

# 配置 CORS（如果需要前端跨域访问）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该配置具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router, prefix="/api/v1", tags=["会议处理"])


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
    
    # uvicorn配置
    # 注意：reload=True 会导致日志输出到子进程，主终端看不到
    # 如果需要看到完整日志，请使用 reload=False
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
