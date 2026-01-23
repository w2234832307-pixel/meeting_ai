# 声纹识别服务配置说明

## 📌 当前状态

**声纹识别功能已设置为可选**，不会影响主服务启动。

### 主要功能（已正常运行）✅
- ✅ 语音转文字（ASR）
- ✅ 文档解析
- ✅ 智能 RAG 检索
- ✅ 结构化输出
- ✅ 知识归档到 Chroma

### 声纹识别功能（可选）⚠️
- 📍 接口：`/api/v1/register_voice`
- 📍 用途：注册员工声纹
- 📍 状态：延迟加载，只有调用时才会尝试加载

---

## 🔧 如果需要声纹识别功能

### 问题原因

声纹识别依赖 `modelscope`，而 `modelscope` 需要特定版本的 `datasets` 库。

当前错误：
```
ImportError: cannot import name 'LargeList' from 'datasets'
```

### 解决方案

#### 方案1：修复 datasets 版本（推荐）

```powershell
# 降级 datasets 到兼容版本
pip install datasets==2.14.0

# 或者升级 modelscope 到最新版本
pip install --upgrade modelscope
```

#### 方案2：完全禁用声纹功能

删除或注释掉 `app/services/voice_service.py` 文件：

```powershell
# 重命名为备份
mv app/services/voice_service.py app/services/voice_service.py.bak
```

#### 方案3：保持现状（推荐）

主服务可以正常运行，声纹接口会返回友好的错误提示。

---

## 🧪 测试声纹功能

### 1. 确保依赖已修复

```powershell
python -c "from modelscope.pipelines import pipeline; print('OK')"
```

### 2. 测试注册接口

```bash
curl -X POST "http://localhost:8001/api/v1/register_voice" \
  -F "file=@voice.wav" \
  -F "employee_id=10001" \
  -F "name=张三"
```

### 3. 预期响应

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

## 📚 相关配置

声纹识别使用相同的 Chroma 配置：

```ini
CHROMA_HOST=192.168.211.74
CHROMA_PORT=8000
CHROMA_COLLECTION_NAME=employee_voice_library
```

---

## ✅ 主服务启动确认

现在启动服务应该会看到：

```
🚀 服务启动成功! 当前模式: API
✅ Embedding服务初始化成功，向量维度: 1024
🔌 Chroma连接成功: 192.168.211.74:8000
✅ 集合 employee_voice_library 已存在，已加载
🔌 监听端口: 8001
```

**不会再有 `ImportError` 错误！**

---

## 🎯 总结

- ✅ **主服务不受影响**：Chroma 迁移完全成功
- ⚠️ **声纹功能可选**：需要时可单独修复依赖
- 📌 **延迟加载**：只有调用声纹接口时才会尝试加载相关依赖

如果不需要声纹识别功能，可以忽略这个文件，主服务已经完全可用！🚀
