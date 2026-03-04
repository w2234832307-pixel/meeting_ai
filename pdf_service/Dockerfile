# pdf_service/Dockerfile
# 使用包含 pdf2htmlEX 的基础镜像 (Alpine版体积小)
FROM pdf2htmlex/pdf2htmlex:0.18.8.rc2-master-20200820-alpine-3.12.0-x86_64

# 安装 Python3 和 pip
RUN apk add --no-cache python3 py3-pip

# 安装 FastAPI 和 Uvicorn
RUN pip3 install --no-cache-dir fastapi uvicorn python-multipart

# 设置工作目录
WORKDIR /app

# 复制 API 代码
COPY main.py .

# 暴露端口
EXPOSE 8000

# 启动服务
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]