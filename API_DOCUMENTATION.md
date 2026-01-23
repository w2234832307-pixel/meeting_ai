# 会议AI服务 - API 接口文档

## 📌 基础信息

- **Base URL**: `http://your-server-ip:8001/api/v1`
- **开发环境**: `http://localhost:8001/api/v1`
- **API文档（Swagger）**: `http://localhost:8001/docs`
- **Content-Type**: `multipart/form-data` 或 `application/json`

---

## 🔑 服务模式

当前支持两种部署模式，通过 `.env` 文件切换：

### 模式1：全三方接口（推荐先部署）
```ini
ASR_SERVICE_TYPE=tencent      # 腾讯云语音识别
LLM_SERVICE_TYPE=api          # DeepSeek API
EMBEDDING_SERVICE=openai      # OpenAI Embedding
```

### 模式2：本地+三方混合
```ini
ASR_SERVICE_TYPE=funasr       # 本地 FunASR 模型
LLM_SERVICE_TYPE=local        # 本地 Qwen3-14b 模型
EMBEDDING_SERVICE=bge-m3      # 本地 BGE-M3 模型
```

**切换方法**：修改 `.env` 文件后，重启服务即可。

---

## 📡 接口列表

### 1. 服务健康检查

#### `GET /`

**描述**：检查服务是否运行

**响应示例**：
```json
{
  "service": "Meeting AI Service",
  "version": "1.0",
  "status": "running"
}
```

---

#### `GET /health`

**描述**：检查服务健康状态和当前配置

**响应示例**：
```json
{
  "status": "healthy",
  "mode": "API"
}
```

---

### 2. 会议处理接口（核心）

#### `POST /api/v1/process`

**描述**：处理音频/文档/文本，生成结构化会议纪要

**支持的输入方式**（5选1）：
1. 上传音频文件
2. 提供音频 URL
3. 提供音频 ID（从数据库获取）
4. 上传文档文件（Word/PDF）
5. 提供纯文本内容

#### 请求参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `file` | File | 否 | 音频文件（mp3/wav/m4a等） |
| `audio_url` | String | 否 | 音频文件 URL（公网可访问） |
| `audio_id` | Integer | 否 | 音频 ID（从数据库获取） |
| `document_file` | File | 否 | 文档文件（docx/pdf/txt） |
| `text_content` | String | 否 | 纯文本内容 |
| `template_id` | String | 是 | 模板 ID（默认: "default"） |

**注意**：以上5个输入参数至少提供1个。

---

#### 请求示例

##### 示例1：上传音频文件

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "file=@meeting.mp3" \
  -F "template_id=default"
```

##### 示例2：提供音频 URL

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "audio_url=https://your-cdn.com/meeting.mp3" \
  -F "template_id=default"
```

##### 示例3：上传 Word 文档

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "document_file=@meeting.docx" \
  -F "template_id=default"
```

##### 示例4：提供纯文本

```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "text_content=今天会议讨论了产品迭代计划，包括新功能开发和性能优化。" \
  -F "template_id=default"
```

##### 示例5：Python 调用

```python
import requests

url = "http://localhost:8001/api/v1/process"

# 方式1：上传音频文件
with open("meeting.mp3", "rb") as f:
    files = {"file": f}
    data = {"template_id": "default"}
    response = requests.post(url, files=files, data=data)
    print(response.json())

# 方式2：提供音频 URL
data = {
    "audio_url": "https://your-cdn.com/meeting.mp3",
    "template_id": "default"
}
response = requests.post(url, data=data)
print(response.json())

# 方式3：纯文本
data = {
    "text_content": "今天会议讨论了产品迭代计划...",
    "template_id": "default"
}
response = requests.post(url, data=data)
print(response.json())
```

---

#### 响应格式

```json
{
  "status": "success",
  "transcript": [
    {
      "text": "大家好，今天我们讨论一下产品迭代计划。",
      "start_time": 0.0,
      "end_time": 3.5,
      "speaker_id": "1"
    },
    {
      "text": "好的，我先介绍一下背景。",
      "start_time": 3.5,
      "end_time": 6.2,
      "speaker_id": "2"
    }
  ],
  "structured_data": "# 会议纪要\n\n## 会议主题\n产品迭代计划讨论\n\n## 关键决策\n1. 确定新功能开发优先级\n2. 性能优化方案评审通过\n\n## 行动项\n- [ ] 张三：完成需求文档（截止：2026-01-25）\n- [ ] 李四：技术方案评审（截止：2026-01-28）",
  "need_rag": true,
  "rag_query": "产品迭代 性能优化",
  "message": "处理成功"
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 处理状态：success / error |
| `transcript` | Array | 逐字稿（仅音频输入有值） |
| `transcript[].text` | String | 文本内容 |
| `transcript[].start_time` | Float | 开始时间（秒） |
| `transcript[].end_time` | Float | 结束时间（秒） |
| `transcript[].speaker_id` | String | 说话人 ID |
| `structured_data` | String | 结构化会议纪要（Markdown 格式） |
| `need_rag` | Boolean | 是否触发了 RAG 检索 |
| `rag_query` | String | RAG 检索关键词（如触发） |
| `message` | String | 处理消息 |

---

### 3. 知识归档接口

#### `POST /api/v1/archive`

**描述**：将最终版会议纪要切片并存入 Chroma 向量数据库，用于后续 RAG 检索

**Content-Type**: `application/json`

#### 请求参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `text` | String | 是 | 最终版会议纪要文本 |
| `source_id` | Integer | 是 | 来源 ID（如数据库中的会议记录ID） |
| `user_id` | Integer | 否 | 用户 ID |
| `meeting_date` | String | 否 | 会议日期 |
| `department` | String | 否 | 部门名称 |

#### 请求示例

```bash
curl -X POST "http://localhost:8001/api/v1/archive" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "# 产品迭代会议纪要\n\n## 会议时间\n2026-01-21\n\n## 参会人员\n张三、李四、王五\n\n## 会议内容\n讨论了新功能开发计划...",
    "source_id": 12345,
    "user_id": 1,
    "meeting_date": "2026-01-21",
    "department": "产品研发部"
  }'
```

#### Python 调用

```python
import requests

url = "http://localhost:8001/api/v1/archive"

data = {
    "text": "# 产品迭代会议纪要\n\n## 会议内容\n...",
    "source_id": 12345,
    "user_id": 1,
    "meeting_date": "2026-01-21",
    "department": "产品研发部"
}

response = requests.post(url, json=data)
print(response.json())
```

#### 响应格式

```json
{
  "status": "success",
  "message": "知识归档成功",
  "source_id": 12345,
  "chunks_count": 8
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 归档状态：success / error |
| `message` | String | 归档消息 |
| `source_id` | Integer | 来源 ID |
| `chunks_count` | Integer | 存储的知识切片数量 |

---

### 4. 声纹注册接口（可选）

#### `POST /api/v1/register_voice`

**描述**：注册员工声纹（需要安装 modelscope 和声纹模型）

**注意**：此接口为可选功能，如未安装相关依赖，会返回友好的错误提示。

#### 请求参数

| 参数名 | 类型 | 必填 | 说明 |
|--------|------|------|------|
| `file` | File | 是 | 语音文件（wav/mp3，建议3-10秒纯语音） |
| `employee_id` | String | 是 | 员工工号 |
| `name` | String | 是 | 员工姓名 |

#### 请求示例

```bash
curl -X POST "http://localhost:8001/api/v1/register_voice" \
  -F "file=@voice.wav" \
  -F "employee_id=10001" \
  -F "name=张三"
```

#### 响应格式

**成功**：
```json
{
  "code": 200,
  "message": "注册成功",
  "data": {
    "employee_id": "10001",
    "name": "张三",
    "vector_dim": 192
  }
}
```

**依赖缺失**：
```json
{
  "code": 500,
  "message": "声纹服务未安装，请联系管理员",
  "data": null
}
```

---

## 🔧 服务切换指南

### 快速切换（修改 .env 即可）

#### 场景1：全三方接口（生产推荐）

**优点**：无需部署模型，成本低，速度快

```ini
# ASR 语音识别
ASR_SERVICE_TYPE=tencent
TENCENT_SECRET_ID=your_id
TENCENT_SECRET_KEY=your_key

# LLM 大语言模型
LLM_SERVICE_TYPE=api
LLM_API_KEY=sk-xxx
LLM_BASE_URL=https://api.deepseek.com

# Embedding 向量化
EMBEDDING_SERVICE=openai
OPENAI_API_KEY=sk-xxx
```

**重启服务**：
```bash
python main.py
```

---

#### 场景2：本地 ASR + 三方 LLM

**优点**：ASR 无限制调用，LLM 保持灵活性

```ini
# ASR 语音识别（本地）
ASR_SERVICE_TYPE=funasr
FUNASR_DEVICE=cuda  # 或 cpu

# LLM 大语言模型（三方）
LLM_SERVICE_TYPE=api
LLM_API_KEY=sk-xxx

# Embedding 向量化（本地）
EMBEDDING_SERVICE=bge-m3
BGE_M3_DEVICE=cuda  # 或 cpu
```

**重启服务**：
```bash
python main.py
```

---

#### 场景3：全本地部署

**优点**：数据隐私，无API限制

```ini
# ASR 语音识别（本地）
ASR_SERVICE_TYPE=funasr
FUNASR_DEVICE=cuda

# LLM 大语言模型（本地）
LLM_SERVICE_TYPE=local
LOCAL_LLM_BASE_URL=http://localhost:8000/v1
LOCAL_LLM_MODEL_NAME=qwen3-14b

# Embedding 向量化（本地）
EMBEDDING_SERVICE=bge-m3
BGE_M3_DEVICE=cuda
```

**重启服务**：
```bash
python main.py
```

---

## 🚀 部署建议

### 第一阶段：快速上线（1小时）

1. **使用全三方接口**
2. **配置 .env**：
   ```ini
   ASR_SERVICE_TYPE=tencent
   LLM_SERVICE_TYPE=api
   EMBEDDING_SERVICE=openai
   ```
3. **启动服务**：`python main.py`
4. **验证接口**：访问 `http://your-ip:8001/docs`

### 第二阶段：优化成本（按需）

1. **部署 FunASR 模型**（如果 ASR 调用频繁）
2. **修改 .env**：
   ```ini
   ASR_SERVICE_TYPE=funasr
   ```
3. **重启服务**

### 第三阶段：完全私有化（可选）

1. **部署 Qwen3-14b LLM**
2. **修改 .env**：
   ```ini
   LLM_SERVICE_TYPE=local
   ```
3. **重启服务**

---

## ⚠️ 注意事项

### 1. 音频 URL 要求

- 腾讯云 ASR（`ASR_SERVICE_TYPE=tencent`）：**必须是公网可访问的 URL**
- 本地 FunASR（`ASR_SERVICE_TYPE=funasr`）：支持本地路径

### 2. 音频时长限制

- 默认限制：5小时（18000秒）
- 配置项：`MAX_AUDIO_DURATION_SECONDS=18000`

### 3. 向量维度

确保 Embedding 服务与 Chroma 中已有数据维度一致：
- BGE-M3: 1024 维
- OpenAI (text-embedding-ada-002): 1536 维
- Tencent NLP: 768 维

---

## 🐛 错误码说明

| HTTP状态码 | 说明 |
|-----------|------|
| 200 | 成功 |
| 400 | 请求参数错误 |
| 500 | 服务器内部错误 |

**错误响应示例**：
```json
{
  "status": "error",
  "transcript": [],
  "structured_data": "",
  "need_rag": false,
  "rag_query": "",
  "message": "错误详情: 音频时长超过限制"
}
```

---

## 📞 联系支持

- **文档**: [QUICK_START.md](QUICK_START.md)
- **切换指南**: [SWITCH_SERVICES.md](SWITCH_SERVICES.md)
- **Chroma 配置**: [CHROMA_MIGRATION.md](CHROMA_MIGRATION.md)

---

## ✅ 快速检查清单

部署前检查：
- [ ] `.env` 文件已配置
- [ ] API Key 已填写（如使用三方接口）
- [ ] 端口 8001 未被占用
- [ ] Chroma 服务器可访问（192.168.211.74:8000）
- [ ] 依赖已安装（`pip install -r requirements.txt`）

服务启动后检查：
- [ ] 访问 `/health` 返回 `healthy`
- [ ] 访问 `/docs` 可以看到 API 文档
- [ ] 测试 `/api/v1/process` 接口返回正常

---

**祝使用愉快！** 🎉
