# FunASR 独立语音识别服务

独立部署的 FunASR 语音识别服务，提供 HTTP API 供其他服务调用。支持高准确率语音识别、说话人分离、声纹识别和热词管理。

## 🎯 功能特点

- ✅ **独立部署**：资源隔离，可被多个服务共享
- ✅ **高准确率识别**：使用 SenseVoiceSmall 模型，识别准确率高
- ✅ **说话人分离**：自动识别不同说话人，支持声纹匹配
- ✅ **时间戳支持**：提供精确的时间戳，支持音频跳转
- ✅ **热词管理**：支持动态热词配置，提升识别准确率
- ✅ **GPU 加速**：支持 CUDA 加速，提升处理速度
- ✅ **HTTP API**：标准 RESTful API，易于集成

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- Windows / Linux / macOS
- 可选：CUDA 11.8+（GPU 加速）

### 2. 创建虚拟环境

```bash
cd funasr_standalone
python -m venv venv

# Windows
.\venv\Scripts\Activate.ps1

# Linux/Mac
source venv/bin/activate
```

### 3. 安装依赖

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

### 4. 配置环境变量（可选）

创建 `.env` 文件（可选，使用默认值也可）：

```ini
# 服务配置
FUNASR_SERVICE_HOST=0.0.0.0
FUNASR_SERVICE_PORT=8002

# 模型配置
FUNASR_DEVICE=cuda  # 或 cpu
FUNASR_NCPU=8       # CPU 核心数（CPU 模式）

# 声纹库配置（可选）
CHROMA_HOST=192.168.211.74
CHROMA_PORT=8000
```

### 5. 启动服务

```bash
python main.py
```

**预期输出**：
```
==================================================
🚀 FunASR 服务启动
📍 地址: http://0.0.0.0:8002
🎤 模型: SenseVoiceSmall
💻 设备: cuda
==================================================
✅ SenseVoiceSmall 加载成功
✅ VAD 模型加载成功
✅ Cam++ 说话人模型加载成功
INFO:     Uvicorn running on http://0.0.0.0:8002
```

### 6. 测试服务

访问 Swagger UI：**http://localhost:8002/docs**

或使用 curl：
```bash
curl http://localhost:8002/health
```

## 📡 API 接口

详细的 API 接口文档请参考：[API_DOCUMENTATION.md](API_DOCUMENTATION.md)

### 主要接口

- **GET /health** - 健康检查
- **POST /transcribe** - 语音识别（支持说话人分离）
- **GET /hotwords** - 获取热词列表
- **POST /hotwords/reload** - 重新加载热词配置

## 🏗️ 项目结构

```
funasr_standalone/
├── main.py                    # FastAPI 应用入口
├── requirements.txt            # Python 依赖
├── hotwords.json              # 热词配置文件
├── audio_preprocessor.py      # 音频预处理服务
├── hotword_service.py         # 热词管理服务
├── speaker_diarization.py     # 说话人分离服务
├── voice_matcher.py           # 声纹匹配服务
└── logs/                      # 日志目录
    └── funasr_service.log     # 服务日志
```

## 🔧 配置说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FUNASR_SERVICE_HOST` | `0.0.0.0` | 服务监听地址 |
| `FUNASR_SERVICE_PORT` | `8002` | 服务端口 |
| `FUNASR_DEVICE` | `cuda` | 设备类型（`cuda`/`cpu`） |
| `FUNASR_NCPU` | `8` | CPU 核心数（CPU 模式） |

### 热词配置

热词配置文件：`hotwords.json`

**配置格式**：
```json
{
  "人名": ["张三", "李四", "王五"],
  "项目名": ["智能办公", "数据中台"],
  "技术词汇": ["机器学习", "深度学习"],
  "mappings": {
    "人名": {
      "小张": "张三",
      "老李": "李四"
    }
  }
}
```

**说明**：
- 支持多个类别，每个类别是一个热词列表
- `mappings` 字段用于名称标准化映射（可选）
- 修改后需要调用 `/hotwords/reload` 重新加载

### 声纹库配置（可选）

如果启用声纹识别，需要配置 ChromaDB：

```ini
CHROMA_HOST=192.168.211.74
CHROMA_PORT=8000
```

声纹库集合名称：`employee_voice_voiceprint`（192维 Cam++ 向量）

## 🎯 使用示例

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
  -F "enable_punc=true" \
  -F "enable_diarization=true"
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
      "speaker_id": 0
    },
    {
      "text": "第二句话",
      "start_time": 2.5,
      "end_time": 5.0,
      "speaker_id": 1
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
    data = {
        "enable_punc": "true",
        "enable_diarization": "true"
    }
    response = requests.post(url, files=files, data=data)
    result = response.json()
    
    print(f"识别文本: {result['text']}")
    print(f"逐字稿: {result['transcript']}")
```

### 4. 获取热词列表

```bash
curl http://localhost:8002/hotwords
```

### 5. 重新加载热词

```bash
curl -X POST http://localhost:8002/hotwords/reload
```

## ⚡ 性能优化

### 1. GPU 加速

确保配置了 GPU：
```ini
FUNASR_DEVICE=cuda
```

### 2. CPU 核心数

CPU 模式下调整核心数：
```ini
FUNASR_NCPU=16  # 根据服务器 CPU 调整
```

### 3. 批处理优化

根据 GPU 显存调整批处理大小（代码中配置）。

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

**A**: 使用 CPU 模式或降低批处理大小：
```ini
FUNASR_DEVICE=cpu
```

### Q4: 说话人ID不连续？

**A**: 系统会自动将说话人ID重新映射为连续编号（0, 1, 2...）。

### Q5: 声纹识别失败？

**A**: 
1. 检查 ChromaDB 连接配置
2. 确保声纹库中有已注册的员工声纹
3. 查看日志文件 `logs/funasr_service.log`

## 📊 性能参考

| 配置 | 处理速度 | 显存占用 | 内存占用 |
|------|----------|----------|----------|
| CPU (8核) | ~0.5x 实时 | - | 4GB |
| GPU (RTX 3060) | ~5x 实时 | 4GB | 2GB |
| GPU (A100) | ~20x 实时 | 6GB | 2GB |

## 🔗 集成到主服务

主服务（`meeting_ai`）的配置（`.env`）：

```ini
# ASR 服务配置
ASR_SERVICE_TYPE=funasr
FUNASR_SERVICE_URL=http://localhost:8002
```

主服务会自动通过 HTTP 调用这个独立服务。

## 📚 相关文档

- [API 接口文档](API_DOCUMENTATION.md)
- [FunASR 官方文档](https://github.com/alibaba-damo-academy/FunASR)
- [主服务部署文档](../README.md)

## 🔍 监控和日志

### 查看日志

```bash
# 服务日志会输出到控制台
python main.py

# 查看日志文件
tail -f logs/funasr_service.log
```

### 性能监控

访问 FastAPI 自带的文档页面：
- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc

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
COPY . .

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

**服务部署完成后，主服务就可以通过 HTTP 调用了！** 🎉
