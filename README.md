# 会议AI服务 (Meeting AI)

一个工业级的会议智能处理系统，支持语音转文字、文档解析、智能摘要、知识库管理和声纹识别。

## ✨ 核心功能

- 🎤 **语音识别**：支持腾讯云ASR / FunASR本地部署，可随时切换，支持说话人分离和逐字时间戳
- 📄 **文档解析**：支持 Word、PDF、TXT 文件自动提取文本
- 🤖 **智能RAG**：自动判断是否需要查询历史知识库，支持语义检索
- 📝 **结构化输出**：基于模板生成会议纪要（Markdown/HTML格式）
- 💾 **知识归档**：Chroma 向量数据库存储，支持语义检索
- ⏱️ **时间戳跳转**：返回带时间戳的逐字稿，前端可实现点击跳转
- 🎯 **说话人识别**：支持声纹识别，自动匹配员工姓名
- 🔄 **灵活切换**：ASR和LLM服务均支持云端API与本地部署无缝切换
- 🔥 **热词管理**：支持动态热词配置，提升识别准确率

## 🚀 快速开始

### 1. 环境要求

- Python 3.10+
- Windows / Linux / macOS

### 2. 安装依赖

```powershell
# 创建虚拟环境（推荐 Python 3.10）
python -m venv venv
.\venv\Scripts\Activate.ps1

# 安装核心依赖（全三方接口，推荐）
pip install -r requirements.txt

# 可选：如需本地 BGE-M3 向量化
# 取消 requirements.txt 中 FlagEmbedding 的注释，然后重新安装

# 可选：如需本地 FunASR 语音识别
# 取消 requirements.txt 中 FunASR 相关依赖的注释，然后重新安装
# 注意：推荐使用独立部署的 FunASR 服务（funasr_standalone），无需安装这些依赖
```

### 3. 配置环境变量

```powershell
# 方法1：自动创建（推荐）
python fix_env.py

# 方法2：手动创建
# 复制 .env.example 为 .env，然后编辑配置
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

# FunASR 服务（如果使用独立部署）
FUNASR_SERVICE_URL=http://localhost:8002
ASR_SERVICE_TYPE=funasr  # 或 tencent
```

### 4. 启动服务

```powershell
python main.py
```

服务启动后访问：**http://localhost:8001/docs** 查看API文档

### 5. 测试服务

访问 Swagger UI：http://localhost:8001/docs

或使用 curl：
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "text_content=今天会议讨论了产品迭代计划" \
  -F "template=default"
```

## 📖 API 文档

详细的 API 接口文档请参考：[API_DOCUMENTATION.md](API_DOCUMENTATION.md)

### 主要接口

- **POST /api/v1/process** - 处理会议音频/文档/文本，生成纪要
- **POST /api/v1/archive** - 归档会议纪要到知识库
- **POST /api/v1/api/voice/register** - 注册员工声纹
- **GET /api/v1/api/hotwords** - 获取热词列表
- **POST /api/v1/api/hotwords/reload** - 重新加载热词

## 🏗️ 项目结构

```
meeting_ai/
├── main.py                        # FastAPI 应用入口
├── requirements.txt               # Python 依赖
├── .env                          # 环境配置（需自行创建）
├── fix_env.py                    # 环境配置生成工具
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
│   ├── prompts/
│   │   └── templates.py          # 提示词模板
│   └── services/
│       ├── asr_factory.py        # ASR 服务工厂
│       ├── tencent_asr.py        # 腾讯云 ASR
│       ├── funasr_service.py     # FunASR 服务客户端
│       ├── llm_factory.py        # LLM 服务工厂
│       ├── llm.py                # LLM 服务（API）
│       ├── local_llm.py          # 本地 LLM 服务
│       ├── vector.py             # 向量数据库服务
│       ├── document.py           # 文档解析服务
│       ├── meeting_history.py    # 历史会议服务
│       ├── prompt_template.py    # 提示词模板服务
│       ├── voice_service.py      # 声纹识别服务
│       └── ...                   # 其他服务
├── logs/                         # 日志目录
└── temp_files/                   # 临时文件目录
```

## 🔧 配置说明

### ASR 服务配置

支持两种 ASR 服务：

1. **腾讯云 ASR**（云端）
   ```ini
   ASR_SERVICE_TYPE=tencent
   TENCENT_SECRET_ID=your_id
   TENCENT_SECRET_KEY=your_key
   ```

2. **FunASR**（本地/独立服务）
   ```ini
   ASR_SERVICE_TYPE=funasr
   FUNASR_SERVICE_URL=http://localhost:8002
   ```

### LLM 服务配置

支持两种 LLM 服务：

1. **API 服务**（DeepSeek/OpenAI）
   ```ini
   LLM_SERVICE_TYPE=api
   LLM_API_KEY=your_key
   LLM_BASE_URL=https://api.deepseek.com
   ```

2. **本地服务**（Qwen3等）
   ```ini
   LLM_SERVICE_TYPE=local
   LOCAL_LLM_BASE_URL=http://localhost:8000/v1
   ```

### Embedding 服务配置

支持多种 Embedding 服务：

1. **BGE-M3**（本地，推荐）
   ```ini
   EMBEDDING_SERVICE=bge-m3
   ```

2. **腾讯云 Embedding**
   ```ini
   EMBEDDING_SERVICE=tencent
   ```

### 向量数据库配置

```ini
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=192.168.211.74
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=employee_voice_library
```

## 🎯 使用示例

### 1. 处理音频文件

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "files=@meeting.mp3" \
  -F "template=default" \
  -F "asr_model=funasr"
```

### 2. 处理文档文件

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "document_file=@meeting.docx" \
  -F "template=default"
```

### 3. 处理纯文本

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "text_content=今天会议讨论了..." \
  -F "template=default"
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

### 5. 注册员工声纹

```bash
curl -X POST "http://localhost:8001/api/v1/api/voice/register" \
  -F "file=@employee_voice.wav" \
  -F "name=张三" \
  -F "employee_id=EMP001"
```

## ⚠️ 重要说明

### 音频处理要求

**腾讯云ASR要求音频文件必须是可公网访问的URL**，不支持本地文件路径。

解决方案：
1. **本地测试**：使用 Python HTTP 服务器
2. **生产环境**：将音频上传到云存储（腾讯云COS、阿里云OSS等）
3. **推荐方案**：使用 FunASR 本地服务，支持直接上传文件

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

### 热词配置

热词配置文件位于 `funasr_standalone/hotwords.json`（如果使用独立 FunASR 服务）。

如果两个服务独立部署，可以通过环境变量指定路径：
```ini
HOTWORDS_JSON_PATH=/path/to/hotwords.json
```

## 🚀 部署说明

### 开发环境 vs 生产环境

- **开发环境**：`http://localhost:8001`（仅本地访问）
- **生产环境**：需要部署到服务器，配置反向代理（Nginx）

### 部署方式

1. **直接部署**：使用 Gunicorn + Nginx
   ```bash
   pip install gunicorn
   gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker -b 0.0.0.0:8001
   ```

2. **Docker 部署**（推荐）：确保环境一致性
3. **云平台**：阿里云、腾讯云等

## 🔑 技术栈

- **Web框架**：FastAPI
- **ASR服务**：腾讯云语音识别 / FunASR本地（可切换）
- **LLM服务**：DeepSeek API / Qwen3本地（可切换）
- **Embedding服务**：BGE-M3本地 / 腾讯云（可切换）
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
**A**: 腾讯云ASR要求URL必须可公网访问，本地路径不支持。建议使用 FunASR 本地服务。

### Q2: 前端同事说访问不到我的服务？
**A**: `localhost` 只能本地访问。需要部署到服务器，给前端提供服务器地址。

### Q3: 如何切换到本地ASR？
**A**: 修改 `.env` 中的 `ASR_SERVICE_TYPE=funasr`，并配置 `FUNASR_SERVICE_URL`。

### Q4: 如何切换到本地LLM？
**A**: 修改 `.env` 中的 `LLM_SERVICE_TYPE=local`，并配置 `LOCAL_LLM_BASE_URL`。

### Q5: 如何添加自定义模板？
**A**: 在 `app/prompts/templates.py` 中添加新模板定义，或通过 API 传入自定义模板。

### Q6: 音频时长限制是多少？
**A**: 最长5小时（18000秒），这是腾讯云API的限制。FunASR 本地服务无此限制。

## 📝 更新日志

### v1.0.0 (2026-01-22)
- ✅ 完成阿里云到腾讯云ASR迁移
- ✅ 添加说话人分离和时间戳支持
- ✅ 添加 Word/PDF 文档解析
- ✅ 实现智能RAG判断
- ✅ 实现模板化输出
- ✅ 添加声纹识别功能
- ✅ 工业级代码重构

## 📞 技术支持

如有问题，请联系开发团队。

---

**项目状态**：✅ 生产就绪 (Production Ready)
