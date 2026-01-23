# 会议AI服务

一个工业级的会议智能处理系统，支持语音转文字、文档解析、智能摘要和知识库管理。

## ✨ 核心功能

- 🎤 **语音识别**：支持腾讯云ASR / FunASR本地部署，可随时切换，支持说话人分离和逐字时间戳
- 📄 **文档解析**：支持 Word、PDF、TXT 文件自动提取文本
- 🤖 **智能RAG**：自动判断是否需要查询历史知识库
- 📝 **结构化输出**：基于模板生成会议纪要（Markdown格式）
- 💾 **知识归档**：Chroma 向量数据库存储，支持语义检索
- ⏱️ **时间戳跳转**：返回带时间戳的逐字稿，前端可实现点击跳转
- 🔄 **灵活切换**：ASR和LLM服务均支持云端API与本地部署无缝切换

## 🚀 快速开始

### 1. 安装依赖

```powershell
# 创建虚拟环境（推荐 Python 3.10）
python -m venv venv
.\venv\Scripts\Activate.ps1

# 安装核心依赖（全三方接口，推荐）
pip install -r requirements.txt

# 可选：如需本地 BGE-M3 向量化
# pip install -r requirements-bge.txt

# 可选：如需本地 FunASR 语音识别
# pip install -r requirements-funasr.txt
```

📖 **详细说明**：[依赖安装指南.md](依赖安装指南.md)

### 2. 配置环境变量

```powershell
# 方法1：自动创建（推荐）
python fix_env.py

# 方法2：手动创建
copy env.example .env
# 用编辑器打开 .env，填写 API Key 等配置
```

**最小配置示例**：
```ini
# 腾讯云 ASR
TENCENT_SECRET_ID=your_secret_id
TENCENT_SECRET_KEY=your_secret_key

# DeepSeek LLM
LLM_API_KEY=your_deepseek_api_key

# BGE-M3 Embedding（本地）
EMBEDDING_SERVICE=bge-m3

# Chroma 向量数据库（公司内部）
CHROMA_HOST=192.168.211.74
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=employee_voice_library
```

### 3. 启动服务

```powershell
python main.py
```

服务启动后访问：http://localhost:8001/docs 查看API文档

---

### 4. 测试服务

访问：http://localhost:8001/docs

或使用 curl：
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "text_content=今天会议讨论了产品迭代计划" \
  -F "template_id=default"
```

## 📖 使用文档

### 🚀 快速开始
- **[快速部署指南.md](快速部署指南.md)** - 10分钟部署全三方接口 ⭐⭐⭐
- **[QUICK_START.md](QUICK_START.md)** - 快速启动指南（5分钟上手）
- **[API_DOCUMENTATION.md](API_DOCUMENTATION.md)** - 完整的 API 接口文档 ⭐
- **[依赖安装指南.md](依赖安装指南.md)** - 依赖安装说明

### 🔄 服务配置
- **[服务切换指南.md](服务切换指南.md)** - 三方接口 ↔️ 本地模型切换 ⭐
- **[SWITCH_SERVICES.md](SWITCH_SERVICES.md)** - 服务切换详细说明
- **[env.cloud.example](env.cloud.example)** - 全三方接口配置模板

### 🛠️ 技术文档
- **[CHROMA_MIGRATION.md](CHROMA_MIGRATION.md)** - Chroma 向量数据库说明
- **[DEPLOYMENT.md](DEPLOYMENT.md)** - 生产环境部署指南
- **[FIX_ENV_ENCODING.md](FIX_ENV_ENCODING.md)** - 编码问题修复指南
- **[VOICE_SERVICE_SETUP.md](VOICE_SERVICE_SETUP.md)** - 声纹识别服务配置
- **[TEST_LOCAL_AUDIO.md](TEST_LOCAL_AUDIO.md)** - 本地音频文件测试指南

## 🎯 API 使用示例

### 1. 处理音频（带时间戳）

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "audio_url=https://your-cdn.com/meeting.mp3" \
  -F "template_id=default"
```

**响应示例**：
```json
{
  "status": "success",
  "transcript": [
    {
      "text": "会议开始",
      "start_time": 0.0,
      "end_time": 1.5,
      "speaker_id": "1"
    }
  ],
  "structured_data": "# 会议纪要\n...",
  "need_rag": true
}
```

### 2. 处理文档（Word/PDF）

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "document_file=@meeting.docx" \
  -F "template_id=default"
```

### 3. 处理纯文本

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "text_content=今天会议讨论了..." \
  -F "template_id=default"
```

### 4. 归档到知识库

```bash
curl -X POST "http://localhost:8001/api/v1/archive" \
  -H "Content-Type: application/json" \
  -d '{
    "minutes_id": 123,
    "markdown_content": "# 会议纪要\n...",
    "user_id": "user_001"
  }'
```

## 🏗️ 项目结构

```
meeting_ai/
├── main.py                        # FastAPI 应用入口
├── requirements.txt               # Python 依赖
├── .env                          # 环境配置（需自行创建）
├── app/
│   ├── api/
│   │   └── endpoints.py          # API 路由和业务逻辑
│   ├── core/
│   │   ├── config.py             # 配置管理
│   │   ├── logger.py             # 日志系统
│   │   ├── exceptions.py         # 自定义异常
│   │   └── utils.py              # 工具函数
│   ├── schemas/
│   │   └── task.py               # 数据模型（Pydantic）
│   └── services/
│       ├── tencent_asr.py        # 腾讯云 ASR
│       ├── llm.py                # LLM 服务
│       ├── vector.py             # 向量数据库
│       ├── document.py           # 文档解析
│       └── ...                   # 其他服务
├── logs/                         # 日志目录
└── temp_files/                   # 临时文件目录
```

## ⚠️ 重要说明

### 音频处理要求

**腾讯云ASR要求音频文件必须是可公网访问的URL**，不支持本地文件路径。

解决方案：
1. **本地测试**：使用 Python HTTP 服务器（详见 [TEST_LOCAL_AUDIO.md](TEST_LOCAL_AUDIO.md)）
2. **生产环境**：将音频上传到云存储（腾讯云COS、阿里云OSS等）

### 时间戳跳转实现

前端可以使用返回的 `transcript` 数据实现点击文本跳转音频：

```javascript
// 点击文本跳转到对应音频位置
function jumpToAudio(audioPlayer, startTime) {
  audioPlayer.currentTime = startTime;
  audioPlayer.play();
}

// 使用示例
transcript.forEach(item => {
  const span = document.createElement('span');
  span.textContent = item.text;
  span.onclick = () => jumpToAudio(audioPlayer, item.start_time);
  document.body.appendChild(span);
});
```

## 🚀 部署说明

### 开发环境 vs 生产环境

- **开发环境**：`http://localhost:8001`（仅本地访问）
- **生产环境**：需要部署到服务器，给前端同事提供服务器地址

### 部署方式

1. **Docker 部署**（推荐）：确保环境一致性
2. **直接部署**：使用 Gunicorn + Nginx
3. **云平台**：阿里云、腾讯云等

详细部署步骤请参考：**[DEPLOYMENT.md](DEPLOYMENT.md)**

## 🔑 技术栈

- **Web框架**：FastAPI
- **ASR服务**：腾讯云语音识别 / FunASR本地（可切换）
- **LLM服务**：DeepSeek API / Qwen3本地（可切换）
- **Embedding服务**：BGE-M3本地 / OpenAI / 腾讯云（可切换）
- **向量数据库**：Chroma
- **文档解析**：python-docx, PyPDF2
- **数据库**：MySQL（可选）

## 📊 代码质量

- ✅ 工业级错误处理和日志系统
- ✅ 完整的配置管理和验证
- ✅ 重试机制和超时控制
- ✅ 类型提示和文档字符串
- ✅ 资源自动清理
- ✅ 单例服务模式

## 🆘 常见问题

### Q1: 音频识别失败："invalid url"
**A**: 腾讯云ASR要求URL必须可公网访问，本地路径不支持。参考 [TEST_LOCAL_AUDIO.md](TEST_LOCAL_AUDIO.md)

### Q2: 前端同事说访问不到我的服务？
**A**: `localhost` 只能本地访问。需要部署到服务器，给前端提供服务器地址。参考 [DEPLOYMENT.md](DEPLOYMENT.md)

### Q3: 如何切换到本地ASR？
**A**: 参考 [FUNASR_DEPLOYMENT.md](FUNASR_DEPLOYMENT.md)，安装FunASR后修改 `.env` 中的 `ASR_SERVICE_TYPE=funasr`

### Q4: 如何切换到本地LLM？
**A**: 参考 [QWEN_LOCAL_DEPLOYMENT.md](QWEN_LOCAL_DEPLOYMENT.md)，部署Qwen3后修改 `.env` 中的 `LLM_SERVICE_TYPE=local`

### Q5: 如何添加自定义模板？
**A**: 在 `app/services/llm.py` 的 `_get_template` 方法中添加新模板定义。

### Q6: 音频时长限制是多少？
**A**: 最长5小时（18000秒），这是腾讯云API的限制。

## 📝 更新日志

### v1.0.0 (2026-01-09)
- ✅ 完成阿里云到腾讯云ASR迁移
- ✅ 添加说话人分离和时间戳支持
- ✅ 添加 Word/PDF 文档解析
- ✅ 实现智能RAG判断
- ✅ 实现模板化输出
- ✅ 工业级代码重构

## 📞 技术支持

如有问题，请联系开发团队。

---

**项目状态**：✅ 生产就绪 (Production Ready)
