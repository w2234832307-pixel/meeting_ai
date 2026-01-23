# FunASR 独立语音识别服务

独立部署的 FunASR 语音识别服务，提供 HTTP API 供其他服务调用。

## 🎯 功能特点

- ✅ 独立部署，资源隔离
- ✅ 支持 GPU 加速
- ✅ HTTP API 接口
- ✅ 可被多个服务共享
- ✅ 易于扩展和维护

---

## 🚀 快速开始

### 1. 创建虚拟环境

```bash
cd funasr_standalone
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 2. 安装依赖

#### CPU 版本（默认）

```bash
pip install -r requirements.txt
```

#### GPU 版本（CUDA 11.8）

```bash
pip install fastapi uvicorn python-multipart funasr modelscope
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### GPU 版本（CUDA 12.1）

```bash
pip install fastapi uvicorn python-multipart funasr modelscope
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 3. 配置环境变量

```bash
# 复制配置文件
cp .env.example .env

# 编辑配置（如果需要）
# Windows: notepad .env
# Linux/Mac: nano .env
```

**配置示例（GPU）**：
```ini
FUNASR_DEVICE=cuda
FUNASR_SERVICE_PORT=8002
```

**配置示例（CPU）**：
```ini
FUNASR_DEVICE=cpu
FUNASR_SERVICE_PORT=8002
FUNASR_NCPU=8
```

### 4. 启动服务

```bash
python main.py
```

**预期输出**：
```
==================================================
🚀 FunASR 服务启动
📍 地址: http://0.0.0.0:8002
🎤 模型: paraformer-zh
💻 设备: cuda
==================================================
✅ FunASR 模型加载成功！设备: cuda
INFO:     Uvicorn running on http://0.0.0.0:8002
```

---

## 📡 API 接口

### 1. 健康检查

```bash
curl http://localhost:8002/health
```

**响应**：
```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

### 2. 语音识别

```bash
curl -X POST "http://localhost:8002/transcribe" \
  -F "file=@audio.mp3" \
  -F "enable_punc=true"
```

**响应**：
```json
{
  "text": "完整的识别文本",
  "transcript": [
    {
      "text": "第一句话",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": "1"
    },
    {
      "text": "第二句话",
      "start_time": 2.5,
      "end_time": 5.0,
      "speaker_id": "1"
    }
  ]
}
```

### 3. Python 调用示例

```python
import requests

url = "http://localhost:8002/transcribe"

with open("audio.mp3", "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)
    result = response.json()
    
    print(f"识别文本: {result['text']}")
    print(f"逐字稿: {result['transcript']}")
```

---

## 🔧 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUNASR_SERVICE_HOST` | 0.0.0.0 | 服务监听地址 |
| `FUNASR_SERVICE_PORT` | 8002 | 服务端口 |
| `FUNASR_DEVICE` | cuda | 设备类型（cuda/cpu） |
| `FUNASR_MODEL_NAME` | paraformer-zh | 模型名称 |
| `FUNASR_NCPU` | 4 | CPU 核心数 |
| `FUNASR_BATCH_SIZE` | 300 | 批处理大小 |

---

## 🐳 Docker 部署（可选）

### 1. 创建 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制代码
COPY main.py .
COPY .env .

# 暴露端口
EXPOSE 8002

# 启动服务
CMD ["python", "main.py"]
```

### 2. 构建镜像

```bash
docker build -t funasr-service:latest .
```

### 3. 运行容器

```bash
# CPU 版本
docker run -d -p 8002:8002 funasr-service:latest

# GPU 版本
docker run -d --gpus all -p 8002:8002 funasr-service:latest
```

---

## 🔍 监控和日志

### 查看日志

```bash
# 服务日志会输出到控制台
python main.py

# Docker 容器日志
docker logs -f <container_id>
```

### 性能监控

访问 FastAPI 自带的文档页面：
- Swagger UI: http://localhost:8002/docs
- ReDoc: http://localhost:8002/redoc

---

## ⚡ 性能优化

### 1. GPU 加速

确保配置了 GPU：
```ini
FUNASR_DEVICE=cuda
```

### 2. 批处理优化

根据 GPU 显存调整批处理大小：
```ini
FUNASR_BATCH_SIZE=500  # 显存大的可以调大
```

### 3. CPU 核心数

CPU 模式下调整核心数：
```ini
FUNASR_NCPU=16  # 根据服务器 CPU 调整
```

---

## 🐛 常见问题

### Q1: 模型下载太慢？

**A**: 设置镜像：
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

### Q2: GPU 不可用？

**A**: 检查 CUDA：
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 内存不足？

**A**: 降低批处理大小或使用 CPU：
```ini
FUNASR_DEVICE=cpu
FUNASR_BATCH_SIZE=100
```

---

## 📊 性能参考

| 配置 | 处理速度 | 显存占用 |
|------|----------|----------|
| CPU (8核) | ~0.5x 实时 | 2GB |
| GPU (RTX 3060) | ~5x 实时 | 4GB |
| GPU (A100) | ~20x 实时 | 6GB |

---

## 🔗 集成到主服务

主服务的配置（`meeting_ai/.env`）：

```ini
# ASR 服务配置
ASR_SERVICE_TYPE=funasr
FUNASR_SERVICE_URL=http://localhost:8002
```

主服务会自动通过 HTTP 调用这个独立服务。

---

## 📚 相关文档

- [FunASR 官方文档](https://github.com/alibaba-damo-academy/FunASR)
- [主服务部署文档](../快速部署指南.md)

---

**服务部署完成后，主服务就可以通过 HTTP 调用了！** 🎉
