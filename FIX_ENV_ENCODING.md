# .env 文件编码问题修复指南

## 🐛 错误原因

Windows环境下，`.env` 文件可能被保存为 GBK 或其他编码，导致 `python-dotenv` 读取失败。

错误信息：
```
UnicodeDecodeError: 'utf-8' codec can't decode bytes in position 926-927: invalid continuation byte
```

---

## ✅ 解决方案（3种方法）

### 方法1：自动修复（推荐）⭐

运行修复脚本（自动检测并转换编码）：

```powershell
python fix_env.py
```

**它会自动：**
- 检测 `.env` 文件的编码（UTF-8/GBK/GB2312）
- 备份原文件为 `.env.backup`
- 转换为 UTF-8 编码
- 验证关键配置

---

### 方法2：手动重建 .env 文件

#### 步骤1：删除旧的 .env 文件

```powershell
# 备份（如果需要）
copy .env .env.backup

# 删除
del .env
```

#### 步骤2：从 env.example 创建新的 .env

```powershell
# 复制
copy env.example .env
```

#### 步骤3：用 VS Code 或 Cursor 编辑 .env

**重要：确保编辑器保存为 UTF-8 编码**

在 VS Code/Cursor 中：
1. 打开 `.env` 文件
2. 右下角点击编码（如 `GBK`）
3. 选择 "通过编码保存" → `UTF-8`
4. 保存文件

---

### 方法3：代码已修复（已完成）✅

我已经修改了 `app/core/config.py`，现在它会自动尝试多种编码：
1. 优先尝试 UTF-8
2. 失败则尝试 GBK（Windows中文）
3. 最后尝试 Latin-1（兜底）

**所以理论上你现在直接运行也应该可以了：**

```powershell
python main.py
```

---

## 🎯 推荐操作流程

### 快速流程（1分钟）

```powershell
# 1. 运行修复脚本
python fix_env.py

# 2. 编辑配置（填写真实的API Key等）
# 用 Cursor 或 VS Code 打开 .env，确保保存为UTF-8

# 3. 启动服务
python main.py
```

---

## 🔍 验证 .env 文件编码

### PowerShell 命令

```powershell
# 查看文件编码信息
Get-Content .env -Encoding UTF8 -TotalCount 5
```

如果报错，说明文件不是 UTF-8。

---

## 💡 预防措施

### 1. 编辑器设置（VS Code/Cursor）

**settings.json**：
```json
{
  "files.encoding": "utf8",
  "files.autoGuessEncoding": true
}
```

### 2. Git 配置（避免提交乱码）

```bash
git config --global core.quotepath false
git config --global i18n.logoutputencoding utf-8
git config --global i18n.commitencoding utf-8
```

---

## 🆘 如果还是报错

### 检查步骤

1. **确认 .env 存在**：
   ```powershell
   ls .env
   ```

2. **查看文件内容**（不要直接cat，可能乱码）：
   ```powershell
   python -c "import pathlib; print(pathlib.Path('.env').read_text(encoding='utf-8'))"
   ```
   
   如果报错，说明不是 UTF-8。

3. **手动创建最小配置**：
   
   删除 `.env`，新建一个只包含核心配置的：
   
   ```ini
   APP_PORT=8001
   ASR_SERVICE_TYPE=tencent
   LLM_SERVICE_TYPE=api
   EMBEDDING_SERVICE=openai
   
   TENCENT_SECRET_ID=your_id
   TENCENT_SECRET_KEY=your_key
   LLM_API_KEY=sk-your_key
   
   CHROMA_HOST=192.168.211.74
   CHROMA_PORT=8000
   CHROMA_COLLECTION_NAME=employee_voice_library
   ```
   
   **用 Cursor 保存为 UTF-8**，然后测试。

---

## 📌 常见问题

### Q: 为什么会出现编码问题？

A: Windows 默认编码是 GBK（中文环境），很多编辑器也默认 GBK。如果 `.env` 中有中文注释，更容易出问题。

### Q: 可以直接用 GBK 编码吗？

A: 可以，我已经修改了代码支持 GBK。但**强烈建议统一用 UTF-8**，避免跨平台问题。

### Q: 我的 .env 没有中文，为什么还报错？

A: 可能是文件被某些工具修改了，或者包含了不可见字符。建议重新创建。

---

## ✅ 完成检查清单

- [ ] 运行 `python fix_env.py` 修复编码
- [ ] 确认 `.env` 文件是 UTF-8 编码
- [ ] 填写真实的 API Key 和配置
- [ ] 运行 `python main.py` 启动服务
- [ ] 访问 http://localhost:8001 验证

---

现在运行 `python fix_env.py`，应该就能解决了！🚀
