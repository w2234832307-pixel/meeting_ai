# Meeting AI API 接口文档

## 基础信息

- **Base URL**: `http://localhost:8001/api/v1`
- **API 版本**: v1
- **文档地址**: http://localhost:8001/docs (Swagger UI)

## 通用说明

### 请求格式

所有接口支持：
- **Content-Type**: `application/json` 或 `multipart/form-data`
- **编码**: UTF-8

### 响应格式

```json
{
  "status": "success|failed|error",
  "message": "提示信息",
  "data": {}
}
```

### 错误码

- `200`: 成功
- `400`: 请求参数错误
- `500`: 服务器内部错误

---

## 1. 会议处理接口

### POST /api/v1/process

处理会议音频/文档/文本，生成会议纪要。

#### 请求参数

##### 输入源（以下7种方式任选其一）

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `files` | File[] | 否 | 音频文件上传（支持 mp3/wav/m4a/mp4 等，支持多个文件） |
| `file_paths` | String | 否 | 本地文件路径（单个：`test_audio/meeting.mp3`，多个：`audio1.mp3,audio2.mp3`） |
| `audio_urls` | String | 否 | 音频URL地址（要求可公网访问，单个或逗号分隔多个） |
| `audio_id` | Integer | 否 | 数据库音频ID（用于处理已存储的历史音频） |
| `document_file` | File | 否 | 文档文件上传（支持 Word .docx / PDF .pdf / 文本 .txt） |
| `text_content` | String | 否 | 纯文本内容（直接输入会议文本，跳过语音识别） |

##### 模板参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `template` | String | 否 | `default` | 模板配置：<br>- 预设模板ID：`default`（标准）/ `simple`（简洁）/ `detailed`（详细）<br>- 文档路径：`D:\模板.docx`（自定义格式）<br>- JSON字符串：自定义提示词<br>- 纯文本：直接的提示词内容 |

##### 用户需求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `user_requirement` | String | 否 | 特殊要求（可选），如"重点关注预算讨论"、"简化技术细节"等 |

##### 历史会议参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `history_meeting_ids` | String | 否 | - | 关联历史会议ID列表（逗号分隔，如：`100,101,102`） |
| `history_mode` | String | 否 | `auto` | 历史处理模式：<br>- `auto`：自动判断（推荐）<br>- `retrieval`：检索模式（查找相关历史内容）<br>- `summary`：总结模式（提供历史会议总结） |

##### 模型配置参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `asr_model` | String | 否 | `auto` | 语音识别模型：<br>- `auto`：自动选择<br>- `funasr`：本地FunASR（推荐）<br>- `tencent`：腾讯云ASR |
| `llm_model` | String | 否 | `auto` | LLM模型选择：<br>- `auto`：自动选择<br>- `deepseek`：DeepSeek API<br>- `qwen3`：本地Qwen3模型 |
| `llm_temperature` | Float | 否 | `0.7` | 生成温度（0.0-1.0）：<br>- `0.3`：更保守，输出更确定<br>- `0.7`：平衡（推荐）<br>- `1.0`：更有创造性 |
| `llm_max_tokens` | Integer | 否 | `2000` | 最大生成长度（token数） |

#### 请求示例

**示例1：上传音频文件**
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "files=@meeting.mp3" \
  -F "template=default" \
  -F "asr_model=funasr"
```

**示例2：使用本地文件路径**
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "file_paths=test_audio/meeting.mp3" \
  -F "template=default"
```

**示例3：使用音频URL（腾讯云ASR）**
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "audio_urls=https://example.com/meeting.mp3" \
  -F "template=default" \
  -F "asr_model=tencent"
```

**示例4：上传文档文件**
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "document_file=@meeting.docx" \
  -F "template=default"
```

**示例5：直接输入文本**
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "text_content=今天会议讨论了产品迭代计划..." \
  -F "template=default"
```

**示例6：带历史会议参考**
```bash
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "files=@meeting.mp3" \
  -F "template=default" \
  -F "history_meeting_ids=100,101,102" \
  -F "history_mode=retrieval"
```

#### 响应示例

```json
{
  "status": "success",
  "message": "处理完成",
  "raw_text": "完整的识别文本...",
  "transcript": [
    {
      "text": "会议开始",
      "start_time": 0.0,
      "end_time": 1.5,
      "speaker_id": 0
    },
    {
      "text": "今天讨论产品迭代",
      "start_time": 1.5,
      "end_time": 5.2,
      "speaker_id": 1
    }
  ],
  "need_rag": true,
  "html_content": "<h1>会议纪要</h1><p>...</p>",
  "usage_tokens": 1500
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 任务状态：`success` / `failed` / `error` |
| `message` | String | 提示信息 |
| `raw_text` | String | 语音转写的原始文本（合并后的完整文本） |
| `transcript` | Array | 带时间戳和说话人的逐字稿 |
| `transcript[].text` | String | 文本内容 |
| `transcript[].start_time` | Float | 开始时间（秒） |
| `transcript[].end_time` | Float | 结束时间（秒） |
| `transcript[].speaker_id` | Integer | 说话人ID（如果启用说话人分离） |
| `need_rag` | Boolean | 是否触发了历史检索 |
| `html_content` | String | HTML格式的纪要 |
| `usage_tokens` | Integer | LLM 消耗的 token 数 |

---

## 2. 归档接口

### POST /api/v1/archive

将会议纪要归档到知识库（向量数据库）。

#### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `minutes_id` | Integer | 是 | MySQL里的会议纪要ID (minutes_draft_id) |
| `markdown_content` | String | 是 | 用户修改确认后的最终版 Markdown 内容 |
| `user_id` | String | 否 | 操作人的ID（可选） |

#### 请求示例

```bash
curl -X POST "http://localhost:8001/api/v1/archive" \
  -H "Content-Type: application/json" \
  -d '{
    "minutes_id": 123,
    "markdown_content": "# 会议纪要\n\n## 会议主题\n...",
    "user_id": "user_001"
  }'
```

#### 响应示例

```json
{
  "status": "success",
  "message": "已成功存入企业知识库",
  "chunks_count": 15
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 状态：`success` / `error` |
| `message` | String | 提示信息 |
| `chunks_count` | Integer | 切分成了多少个片段存入向量库 |

---

## 3. 声纹注册接口

### POST /api/v1/api/voice/register

注册员工声纹，用于说话人识别。

#### 请求参数

| 参数 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `file` | File | 是 | 员工录音文件（wav/mp3） |
| `name` | String | 是 | 员工姓名 |
| `employee_id` | String | 是 | 员工工号（唯一标识） |

#### 请求示例

```bash
curl -X POST "http://localhost:8001/api/v1/api/voice/register" \
  -F "file=@employee_voice.wav" \
  -F "name=张三" \
  -F "employee_id=EMP001"
```

#### 响应示例

**成功**
```json
{
  "code": 200,
  "message": "注册成功",
  "data": {
    "employee_id": "EMP001",
    "name": "张三",
    "vector_dim": 192
  }
}
```

**失败**
```json
{
  "code": 400,
  "message": "音频质量过差或过短，无法提取声纹特征，请重录",
  "data": null
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | Integer | 状态码：200（成功）/ 400（失败）/ 500（错误） |
| `message` | String | 提示信息 |
| `data.employee_id` | String | 员工工号 |
| `data.name` | String | 员工姓名 |
| `data.vector_dim` | Integer | 声纹向量维度（通常为192） |

---

## 4. 热词管理接口

### GET /api/v1/api/hotwords

获取当前热词列表（转发到FunASR服务）。

#### 请求示例

```bash
curl -X GET "http://localhost:8001/api/v1/api/hotwords"
```

#### 响应示例

```json
{
  "code": 200,
  "message": "获取成功",
  "data": {
    "categories": ["人名", "项目名", "技术词汇"],
    "hotwords": {
      "人名": ["张三", "李四"],
      "项目名": ["智能办公", "数据中台"]
    },
    "stats": {
      "人名": 2,
      "项目名": 2
    },
    "total": 4
  }
}
```

---

### POST /api/v1/api/hotwords/reload

重新加载热词配置（转发到FunASR服务，用于修改 `funasr_standalone/hotwords.json` 后刷新）。

#### 请求示例

```bash
curl -X POST "http://localhost:8001/api/v1/api/hotwords/reload"
```

#### 响应示例

```json
{
  "code": 200,
  "message": "热词重载成功",
  "data": {
    "total": 4,
    "stats": {
      "人名": 2,
      "项目名": 2
    }
  }
}
```

---

## 错误处理

### 错误响应格式

```json
{
  "status": "error",
  "message": "错误描述信息"
}
```

### 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| 400 | 请求参数错误 | 检查请求参数格式和必填项 |
| 500 | 服务器内部错误 | 查看服务器日志，联系技术支持 |
| 503 | 服务不可用 | 检查 ASR/LLM 服务是否正常运行 |

---

## 注意事项

1. **音频文件要求**：
   - 腾讯云ASR要求音频URL必须可公网访问
   - FunASR支持直接上传文件
   - 支持格式：mp3, wav, m4a, mp4 等

2. **文件大小限制**：
   - 默认最大文件大小：500MB
   - 可通过环境变量 `MAX_FILE_SIZE_MB` 配置

3. **音频时长限制**：
   - 腾讯云ASR：最长5小时（18000秒）
   - FunASR：无限制

4. **超时设置**：
   - ASR超时：默认2小时（7200秒）
   - LLM超时：默认3分钟（180秒）
   - 可通过环境变量配置

5. **热词配置**：
   - 热词配置文件位于 `funasr_standalone/hotwords.json`
   - 修改后需要调用 `/api/v1/api/hotwords/reload` 重新加载

---

## 更多信息

- **Swagger UI**: http://localhost:8001/docs
- **ReDoc**: http://localhost:8001/redoc
- **项目 README**: [README.md](README.md)
