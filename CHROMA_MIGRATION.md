# Milvus → Chroma 迁移完成指南

## ✅ 已完成的修改

### 1. 代码修改

#### `app/services/vector.py`
- ✅ 将 Milvus 客户端替换为 Chroma HttpClient
- ✅ 使用 `chromadb.HttpClient()` 连接到远程 Chroma 服务器
- ✅ 集合初始化逻辑适配 Chroma API
- ✅ `search_similar()` 方法使用 Chroma 的 `query()` 接口
- ✅ `save_knowledge()` 方法使用 Chroma 的 `add()` 接口
- ✅ 距离计算改为 L2 距离转相似度

#### `app/core/config.py`
- ✅ 删除 Milvus 配置（`MILVUS_HOST`, `MILVUS_PORT`）
- ✅ 添加 Chroma 配置（`VECTOR_STORE_TYPE`, `CHROMA_HOST`, `CHROMA_PORT`, `CHROMA_COLLECTION_NAME`）
- ✅ 默认值设置为公司内部 Chroma 服务器（192.168.211.74:8000）

#### `requirements.txt`
- ✅ 删除 `pymilvus==2.3.6`
- ✅ 添加 `chromadb==0.4.22`

#### `env.example`
- ✅ 更新向量数据库配置示例为 Chroma

---

## 🚀 如何使用

### 1. 安装依赖

```powershell
# 卸载旧的 Milvus 客户端（如果已安装）
pip uninstall pymilvus -y

# 安装 Chroma 客户端
pip install chromadb==0.4.22
```

### 2. 配置 .env 文件

确保你的 `.env` 文件包含以下配置：

```ini
# --- 向量数据库配置（Chroma）---
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=192.168.211.74
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=employee_voice_library
```

### 3. 启动服务

```powershell
python main.py
```

---

## 🔍 Chroma vs Milvus 对比

| 功能             | Milvus                          | Chroma                          |
|------------------|--------------------------------|---------------------------------|
| **连接方式**     | `connections.connect()`        | `chromadb.HttpClient()`         |
| **集合获取**     | `Collection(name)`             | `client.get_collection(name)`   |
| **集合创建**     | `Collection(name, schema)`     | `client.create_collection(name)`|
| **数据插入**     | `collection.insert([data])`    | `collection.add(ids, embeddings, documents, metadatas)` |
| **相似度搜索**   | `collection.search()`          | `collection.query()`            |
| **距离度量**     | COSINE, L2, IP                 | L2（默认）                      |
| **返回格式**     | `hits` with `distance`         | `documents`, `distances`, `metadatas` |

---

## 🛠️ 技术细节

### 相似度计算

**Milvus（COSINE）**：
- 返回余弦相似度，范围 0-1
- 值越大越相似
- 阈值：`similarity > min_score`

**Chroma（L2）**：
- 返回 L2 距离（欧几里得距离）
- 值越小越相似
- 转换公式：`similarity = 1 / (1 + distance)`
- 阈值：`similarity > min_score`

### 元数据存储

**Milvus**：
- 元数据存为 JSON 字符串（VARCHAR 字段）
- 需要手动 `json.dumps()` 和 `json.loads()`

**Chroma**：
- 元数据直接存为字典
- 自动序列化/反序列化

### 批量插入

**Milvus**：
```python
insert_data = [
    embeddings_batch,  # 向量列表
    texts_batch,       # 文本列表
    metadata_batch     # 元数据列表（JSON字符串）
]
collection.insert(insert_data)
collection.flush()
```

**Chroma**：
```python
collection.add(
    ids=ids_batch,              # 唯一ID列表
    embeddings=embeddings_batch, # 向量列表
    documents=documents_batch,   # 文本列表
    metadatas=metadatas_batch    # 元数据列表（字典）
)
```

---

## 📋 清理工作

### 已删除的配置

- `MILVUS_HOST`
- `MILVUS_PORT`

### 已删除的依赖

- `pymilvus==2.3.6`

### 不需要的服务

如果你之前在本地运行了 Milvus，可以停止并卸载：

```bash
# 停止 Milvus（如果用 Docker）
docker stop milvus-standalone
docker rm milvus-standalone

# 删除数据卷（可选，小心！）
docker volume rm milvus-etcd milvus-minio milvus-data
```

---

## 🔗 连接到公司 Chroma

你的代码现在会连接到：

- **主机**: `192.168.211.74`
- **端口**: `8000`
- **集合**: `employee_voice_library`

### 验证连接

```python
import chromadb

client = chromadb.HttpClient(
    host="192.168.211.74",
    port=8000
)

# 测试心跳
client.heartbeat()

# 获取集合
collection = client.get_collection("employee_voice_library")
print(f"集合记录数: {collection.count()}")
```

---

## ⚠️ 注意事项

### 1. 网络访问

确保你的开发机器能访问 `192.168.211.74:8000`：

```powershell
# Windows 测试连接
Test-NetConnection -ComputerName 192.168.211.74 -Port 8000

# 或者用 curl
curl http://192.168.211.74:8000/api/v1/heartbeat
```

### 2. 集合名称

集合名称必须与 Chroma 服务器上已存在的集合名称一致：
- 配置中：`CHROMA_COLLECTION_NAME=employee_voice_library`
- 如果集合不存在，代码会自动创建

### 3. 向量维度

确保你的 Embedding 服务生成的向量维度与 Chroma 中已有的数据一致。
- BGE-M3: 1024 维
- OpenAI (text-embedding-ada-002): 1536 维
- Tencent NLP: 768 维

---

## 🎯 测试检查清单

- [ ] 运行 `python fix_env.py` 修复编码问题
- [ ] 运行 `pip install chromadb==0.4.22`
- [ ] 确认 `.env` 配置正确
- [ ] 测试网络连接到 Chroma 服务器
- [ ] 启动服务 `python main.py`
- [ ] 验证日志显示 "Chroma连接成功"
- [ ] 测试 `/api/v1/process` 接口
- [ ] 测试 `/api/v1/archive` 接口

---

## ✅ 迁移完成！

现在你的系统已经从 Milvus 切换到 Chroma。所有向量检索功能都通过公司内部的 Chroma 服务器（`192.168.211.74:8000`）。

如有问题，检查：
1. 网络连接
2. Chroma 服务器状态
3. 集合名称是否正确
4. Embedding 向量维度是否匹配

🚀 开始使用吧！
