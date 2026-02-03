# FunASR 独立服务 API 接口文档

## 基础信息

- **Base URL**: `http://localhost:8002`
- **API 版本**: v1
- **文档地址**: http://localhost:8002/docs (Swagger UI)

## 通用说明

### 请求格式

所有接口支持：
- **Content-Type**: `application/json` 或 `multipart/form-data`
- **编码**: UTF-8

### 响应格式

```json
{
  "code": 0,
  "msg": "success",
  "data": {}
}
```

### 错误码

- `0`: 成功
- `500`: 服务器内部错误

---

## 1. 健康检查接口

### GET /health

检查服务健康状态和模型加载情况。

#### 请求示例

```bash
curl http://localhost:8002/health
```

#### 响应示例

```json
{
  "status": "healthy",
  "model_loaded": true,
  "device": "cuda"
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | String | 服务状态：`healthy` |
| `model_loaded` | Boolean | 模型是否已加载 |
| `device` | String | 设备类型：`cuda` / `cpu` |

---

## 2. 语音识别接口

### POST /transcribe

进行语音识别，支持说话人分离、时间戳和声纹匹配。

#### 请求参数

| 参数 | 类型 | 必填 | 默认值 | 说明 |
|------|------|------|--------|------|
| `file` | File | 是 | - | 音频文件（支持 mp3/wav/m4a/mp4 等） |
| `enable_punc` | String | 否 | `true` | 是否启用标点符号 |
| `enable_diarization` | String | 否 | `true` | 是否启用说话人分离 |
| `hotword` | String | 否 | - | 外部传入的热词（可选，会与配置文件中的热词合并） |

#### 请求示例

**示例1：基础识别**
```bash
curl -X POST "http://localhost:8002/transcribe" \
  -F "file=@audio.mp3"
```

**示例2：带说话人分离**
```bash
curl -X POST "http://localhost:8002/transcribe" \
  -F "file=@audio.mp3" \
  -F "enable_diarization=true"
```

**示例3：带热词**
```bash
curl -X POST "http://localhost:8002/transcribe" \
  -F "file=@audio.mp3" \
  -F "hotword=张三 李四 智能办公"
```

#### 响应示例

```json
{
  "text": "完整的识别文本",
  "transcript": [
    {
      "text": "第一句话",
      "start_time": 0.0,
      "end_time": 2.5,
      "speaker_id": 0,
      "speaker_name": "张三"
    },
    {
      "text": "第二句话",
      "start_time": 2.5,
      "end_time": 5.0,
      "speaker_id": 1,
      "speaker_name": "李四"
    }
  ]
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `text` | String | 完整的识别文本（合并所有片段） |
| `transcript` | Array | 带时间戳和说话人的逐字稿 |
| `transcript[].text` | String | 文本内容 |
| `transcript[].start_time` | Float | 开始时间（秒） |
| `transcript[].end_time` | Float | 结束时间（秒） |
| `transcript[].speaker_id` | Integer | 说话人ID（连续编号：0, 1, 2...） |
| `transcript[].speaker_name` | String | 说话人姓名（如果声纹匹配成功） |

#### 注意事项

1. **说话人ID**：系统会自动将说话人ID重新映射为连续编号（0, 1, 2...），确保ID连续且从0开始。

2. **声纹匹配**：
   - 如果声纹库中有匹配的声纹，会返回 `speaker_name`
   - 如果未匹配到，`speaker_name` 可能为 `null` 或不包含该字段

3. **音频格式**：支持常见音频格式（mp3, wav, m4a, mp4 等），会自动转换。

4. **处理时间**：根据音频长度和设备性能，处理时间会有所不同。

---

## 3. 热词管理接口

### GET /hotwords

获取当前热词列表和统计信息。

#### 请求示例

```bash
curl http://localhost:8002/hotwords
```

#### 响应示例

```json
{
  "code": 0,
  "msg": "success",
  "data": {
    "categories": ["人名", "项目名", "技术词汇"],
    "hotwords": {
      "人名": ["张三", "李四", "王五"],
      "项目名": ["智能办公", "数据中台"],
      "技术词汇": ["机器学习", "深度学习"]
    },
    "stats": {
      "人名": 3,
      "项目名": 2,
      "技术词汇": 2
    },
    "total": 7
  }
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | Integer | 状态码：0（成功）/ 500（错误） |
| `msg` | String | 提示信息 |
| `data.categories` | Array | 热词类别列表 |
| `data.hotwords` | Object | 各类别的热词列表 |
| `data.stats` | Object | 各类别的热词数量统计 |
| `data.total` | Integer | 热词总数 |

---

### POST /hotwords/reload

重新加载热词配置文件（`hotwords.json`）。

**使用场景**：修改 `hotwords.json` 文件后，调用此接口重新加载，无需重启服务。

#### 请求示例

```bash
curl -X POST http://localhost:8002/hotwords/reload
```

#### 响应示例

**成功**
```json
{
  "code": 0,
  "msg": "热词重载成功",
  "data": {
    "total": 7,
    "stats": {
      "人名": 3,
      "项目名": 2,
      "技术词汇": 2
    }
  }
}
```

**失败**
```json
{
  "code": 500,
  "msg": "重载失败"
}
```

#### 响应字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `code` | Integer | 状态码：0（成功）/ 500（错误） |
| `msg` | String | 提示信息 |
| `data.total` | Integer | 重载后的热词总数 |
| `data.stats` | Object | 各类别的热词数量统计 |

---

## 错误处理

### 错误响应格式

```json
{
  "code": 500,
  "msg": "错误描述信息"
}
```

### 常见错误

| 错误码 | 说明 | 解决方案 |
|--------|------|----------|
| 500 | 服务器内部错误 | 查看服务器日志 `logs/funasr_service.log` |
| 400 | 请求参数错误 | 检查请求参数格式 |

---

## 使用示例

### Python 示例

```python
import requests

# 1. 健康检查
response = requests.get("http://localhost:8002/health")
print(response.json())

# 2. 语音识别
with open("audio.mp3", "rb") as f:
    files = {"file": f}
    data = {
        "enable_punc": "true",
        "enable_diarization": "true"
    }
    response = requests.post(
        "http://localhost:8002/transcribe",
        files=files,
        data=data
    )
    result = response.json()
    print(f"识别文本: {result['text']}")
    print(f"逐字稿: {result['transcript']}")

# 3. 获取热词
response = requests.get("http://localhost:8002/hotwords")
print(response.json())

# 4. 重新加载热词
response = requests.post("http://localhost:8002/hotwords/reload")
print(response.json())
```

### JavaScript 示例

```javascript
// 1. 健康检查
fetch('http://localhost:8002/health')
  .then(res => res.json())
  .then(data => console.log(data));

// 2. 语音识别
const formData = new FormData();
formData.append('file', audioFile);
formData.append('enable_diarization', 'true');

fetch('http://localhost:8002/transcribe', {
  method: 'POST',
  body: formData
})
  .then(res => res.json())
  .then(data => {
    console.log('识别文本:', data.text);
    console.log('逐字稿:', data.transcript);
  });
```

---

## 注意事项

1. **音频文件大小**：建议单个文件不超过 500MB。

2. **处理时间**：
   - CPU 模式：约 0.5x 实时（1小时音频需要约2小时处理）
   - GPU 模式：约 5-20x 实时（取决于GPU性能）

3. **热词配置**：
   - 配置文件：`hotwords.json`
   - 修改后需要调用 `/hotwords/reload` 重新加载
   - 热词会与外部传入的 `hotword` 参数合并

4. **说话人分离**：
   - 说话人ID会自动重新映射为连续编号（0, 1, 2...）
   - 如果启用声纹匹配，会尝试匹配员工姓名

5. **声纹识别**：
   - 需要配置 ChromaDB 连接
   - 声纹库集合：`employee_voice_voiceprint`（192维）
   - 如果声纹库为空或未配置，说话人分离仍可正常工作，但不会返回姓名

---

## 更多信息

- **Swagger UI**: http://localhost:8002/docs
- **ReDoc**: http://localhost:8002/redoc
- **项目 README**: [README.md](README.md)
