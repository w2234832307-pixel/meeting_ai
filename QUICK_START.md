# 快速启动指南

## 📋 前置条件

- Python 3.10 或 3.11（推荐 3.10）
- 公司内部 Chroma 服务器访问权限（192.168.211.74:8000）

---

## 🚀 快速开始（5分钟）

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd meeting_ai
```

### 2. 创建虚拟环境

```powershell
# Windows PowerShell
python -m venv venv
.\venv\Scripts\Activate.ps1

# 如果遇到权限问题
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. 安装依赖

```powershell
pip install -r requirements.txt
```

### 4. 配置环境变量

#### 方法1：自动修复（推荐）

```powershell
# 运行修复脚本（会从 env.example 创建 .env）
python fix_env.py
```

#### 方法2：手动创建

```powershell
# 复制示例配置
copy env.example .env

# 用编辑器打开 .env，填写以下关键配置
```

**最小配置（.env）**：

```ini
# 应用端口
APP_PORT=8001

# ASR服务（腾讯云）
ASR_SERVICE_TYPE=tencent
TENCENT_SECRET_ID=your_tencent_secret_id
TENCENT_SECRET_KEY=your_tencent_secret_key

# LLM服务（DeepSeek API）
LLM_SERVICE_TYPE=api
LLM_API_KEY=your_deepseek_api_key
LLM_BASE_URL=https://api.deepseek.com
LLM_MODEL_NAME=deepseek-chat

# Embedding服务（BGE-M3本地）
EMBEDDING_SERVICE=bge-m3

# 向量数据库（公司内部Chroma）
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=192.168.211.74
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=employee_voice_library
```

### 5. 启动服务

```powershell
python main.py
```

看到以下日志表示启动成功：

```
🚀 服务启动成功! 当前模式: API
🔌 监听端口: 8001
✅ Embedding服务初始化成功，向量维度: 1024
🔌 Chroma连接成功: 192.168.211.74:8000
```

### 6. 测试接口

访问：http://localhost:8001/docs

---

## 📝 API 使用示例

### 1. 处理音频文件（语音转文字 + 结构化）

```python
import requests

url = "http://localhost:8001/api/v1/process"

# 方法1：上传本地音频文件
with open("test_audio/meeting.mp3", "rb") as f:
    files = {"file": f}
    data = {"template_id": "default"}
    response = requests.post(url, files=files, data=data)

# 方法2：提供音频URL
data = {
    "audio_url": "https://example.com/audio.mp3",
    "template_id": "default"
}
response = requests.post(url, data=data)

print(response.json())
```

### 2. 处理纯文本

```python
data = {
    "text_content": "今天会议讨论了...",
    "template_id": "default"
}
response = requests.post(url, data=data)
print(response.json())
```

### 3. 归档知识到 Chroma

```python
url = "http://localhost:8001/api/v1/archive"

data = {
    "text": "最终版会议纪要内容...",
    "source_id": 12345,
    "user_id": 1
}
response = requests.post(url, json=data)
print(response.json())
```

---

## 🔧 服务切换

### 切换 ASR 服务

```ini
# 使用腾讯云（需要API Key）
ASR_SERVICE_TYPE=tencent

# 使用本地 FunASR（需要先部署模型）
ASR_SERVICE_TYPE=funasr
```

### 切换 LLM 服务

```ini
# 使用 DeepSeek API
LLM_SERVICE_TYPE=api
LLM_API_KEY=sk-xxx

# 使用本地 Qwen3-14b（需要先部署）
LLM_SERVICE_TYPE=local
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
```

### 切换 Embedding 服务

```ini
# 使用本地 BGE-M3（推荐，免费）
EMBEDDING_SERVICE=bge-m3

# 使用 OpenAI API
EMBEDDING_SERVICE=openai
OPENAI_API_KEY=sk-xxx

# 使用腾讯云
EMBEDDING_SERVICE=tencent
TENCENT_NLP_SECRET_ID=xxx
TENCENT_NLP_SECRET_KEY=xxx
```

---

## 🐛 常见问题

### Q1: 编码错误（UnicodeDecodeError）

**A**: 运行修复脚本：

```powershell
python fix_env.py
```

### Q2: Chroma 连接失败

**A**: 检查网络连接：

```powershell
# 测试连接
Test-NetConnection -ComputerName 192.168.211.74 -Port 8000

# 或者
curl http://192.168.211.74:8000/api/v1/heartbeat
```

### Q3: BGE-M3 下载太慢

**A**: 使用镜像加速：

```bash
# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# Windows PowerShell
$env:HF_ENDPOINT="https://hf-mirror.com"
```

或者临时切换到云端 Embedding：

```ini
EMBEDDING_SERVICE=openai
OPENAI_API_KEY=sk-xxx
```

### Q4: 依赖安装失败（numpy/pandas）

**A**: 确保使用 Python 3.10 或 3.11：

```powershell
python --version  # 应该显示 3.10.x 或 3.11.x

# 如果版本不对，重新创建虚拟环境
python3.10 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

---

## 📚 更多文档

- [服务切换指南](SWITCH_SERVICES.md)
- [Chroma 迁移说明](CHROMA_MIGRATION.md)
- [部署指南](DEPLOYMENT.md)
- [编码问题修复](FIX_ENV_ENCODING.md)

---

## ✅ 启动检查清单

- [ ] Python 版本正确（3.10 或 3.11）
- [ ] 虚拟环境已激活
- [ ] 依赖已安装（`pip install -r requirements.txt`）
- [ ] `.env` 文件已配置（至少填写 API Key）
- [ ] 能访问 Chroma 服务器（192.168.211.74:8000）
- [ ] 服务启动成功（`python main.py`）
- [ ] 访问 http://localhost:8001/docs 正常

---

🎉 **开始使用吧！**
