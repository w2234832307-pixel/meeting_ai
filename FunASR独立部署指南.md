# FunASR 独立部署指南

## 🎯 架构说明

将 FunASR 模型部署为独立服务，主服务通过 HTTP 调用。

```
┌─────────────────────────────────┐
│   FunASR 独立服务                │
│   (GPU 服务器)                   │
│   端口: 8002                     │
│   依赖: funasr, torch            │
└─────────────────────────────────┘
                ↑
                │ HTTP 调用
                │
┌─────────────────────────────────┐
│   会议AI主服务                    │
│   (轻量服务器)                    │
│   端口: 8001                     │
│   无需安装 funasr!               │
└─────────────────────────────────┘
```

---

## ✅ 优势

1. **资源分离**：FunASR 部署在有 GPU 的机器，主服务可以部署在轻量机器
2. **依赖隔离**：主服务不需要安装 PyTorch 等大依赖（节省 ~5GB）
3. **资源共享**：多个服务可以共用一个 FunASR 服务
4. **易于维护**：可以独立升级、重启 FunASR 服务

---

## 📦 部署步骤

### 第一步：部署 FunASR 独立服务

#### 1.1 进入 FunASR 目录

```bash
cd funasr_standalone
```

#### 1.2 创建虚拟环境

```powershell
# Windows
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

#### 1.3 安装依赖

**CPU 版本**：
```bash
pip install -r requirements.txt
```

**GPU 版本（推荐，CUDA 11.8）**：
```bash
pip install fastapi uvicorn python-multipart funasr modelscope
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**GPU 版本（CUDA 12.1）**：
```bash
pip install fastapi uvicorn python-multipart funasr modelscope
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 1.4 配置环境变量

```bash
# 复制配置文件
cp env.example .env

# 编辑配置
# Windows: notepad .env
# Linux/Mac: nano .env
```

**配置示例（GPU）**：
```ini
FUNASR_DEVICE=cuda
FUNASR_SERVICE_PORT=8002
FUNASR_NCPU=8
FUNASR_BATCH_SIZE=500
```

**配置示例（CPU）**：
```ini
FUNASR_DEVICE=cpu
FUNASR_SERVICE_PORT=8002
FUNASR_NCPU=16
FUNASR_BATCH_SIZE=300
```

#### 1.5 启动服务

```bash
python main.py
```

**预期输出**：
```
==================================================
🚀 FunASR 服务启动
📍 地址: http://0.0.0.0:8002
🎤 模型: paraformer-zh
💻 设备: cuda
==================================================
✅ FunASR 模型加载成功！设备: cuda
INFO:     Uvicorn running on http://0.0.0.0:8002
```

#### 1.6 验证服务

```bash
# 健康检查
curl http://localhost:8002/health

# 测试识别
curl -X POST "http://localhost:8002/transcribe" \
  -F "file=@test.mp3"
```

---

### 第二步：配置主服务

#### 2.1 回到主项目目录

```bash
cd ..  # 回到 meeting_ai 目录
```

#### 2.2 修改 .env 配置

```ini
# ASR 服务配置
ASR_SERVICE_TYPE=funasr
FUNASR_SERVICE_URL=http://localhost:8002

# 注意：主服务不需要安装 funasr 相关依赖！
```

**如果 FunASR 服务在其他机器**：
```ini
FUNASR_SERVICE_URL=http://192.168.1.100:8002
```

#### 2.3 启动主服务

```bash
# 主服务只需要基础依赖
pip install -r requirements.txt

# 启动
python main.py
```

**预期输出**：
```
🚀 服务启动成功! 当前模式: API
🔧 ASR服务类型: funasr
🌐 FunASR 服务模式: HTTP (http://localhost:8002)
✅ FunASR 服务连接成功: cuda
```

---

## 🔄 两种模式对比

### 模式1：HTTP 调用（推荐）⭐

**配置**：
```ini
ASR_SERVICE_TYPE=funasr
FUNASR_SERVICE_URL=http://localhost:8002
```

**优点**：
- ✅ 主服务不需要安装 funasr
- ✅ 可以部署在不同机器
- ✅ 资源共享
- ✅ 易于维护

**主服务依赖**：只需 `requirements.txt`（~500MB）

---

### 模式2：本地加载（不推荐）

**配置**：
```ini
ASR_SERVICE_TYPE=funasr
# 不配置 FUNASR_SERVICE_URL
FUNASR_DEVICE=cuda
```

**缺点**：
- ❌ 主服务需要安装 funasr + torch（~5GB）
- ❌ 启动慢（需要加载模型）
- ❌ 资源占用大

**主服务依赖**：需要 `requirements-funasr.txt`（~5GB）

---

## 📊 资源对比

| 项目 | HTTP 模式 | 本地模式 |
|------|----------|----------|
| **主服务依赖** | 500MB | 5GB |
| **主服务启动时间** | 5秒 | 30秒+ |
| **GPU 要求** | 无 | 需要 |
| **可扩展性** | 高 | 低 |
| **维护难度** | 低 | 高 |

---

## 🚀 生产部署建议

### 方案1：单机部署（开发/测试）

```
同一台机器:
- FunASR 服务: 端口 8002
- 主服务: 端口 8001
```

**配置**：
```ini
FUNASR_SERVICE_URL=http://localhost:8002
```

---

### 方案2：分离部署（生产推荐）⭐

```
GPU 服务器 (192.168.1.100):
- FunASR 服务: 端口 8002

应用服务器 (192.168.1.101):
- 主服务: 端口 8001
```

**FunASR 服务器配置**：
```ini
FUNASR_SERVICE_HOST=0.0.0.0  # 允许外部访问
FUNASR_SERVICE_PORT=8002
FUNASR_DEVICE=cuda
```

**主服务器配置**：
```ini
FUNASR_SERVICE_URL=http://192.168.1.100:8002
```

---

### 方案3：负载均衡（高并发）

```
Nginx 负载均衡:
- FunASR 服务1: 192.168.1.100:8002
- FunASR 服务2: 192.168.1.101:8002
- FunASR 服务3: 192.168.1.102:8002

主服务:
- FUNASR_SERVICE_URL=http://nginx-lb:8002
```

---

## 🐳 Docker 部署

### FunASR 服务 Dockerfile

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 复制文件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
COPY .env .

EXPOSE 8002

CMD ["python", "main.py"]
```

### 启动命令

```bash
# 构建镜像
docker build -t funasr-service:latest .

# 运行（GPU）
docker run -d --gpus all -p 8002:8002 funasr-service:latest

# 运行（CPU）
docker run -d -p 8002:8002 funasr-service:latest
```

---

## 🔍 监控和维护

### 健康检查

```bash
# 定期检查服务状态
curl http://localhost:8002/health
```

### 日志查看

```bash
# FunASR 服务日志
cd funasr_standalone
python main.py  # 查看控制台输出

# 或使用 Docker
docker logs -f funasr-service
```

### 性能监控

访问 API 文档：
- http://localhost:8002/docs

---

## 🐛 常见问题

### Q1: 主服务连接不到 FunASR 服务？

**A**: 检查：
1. FunASR 服务是否启动：`curl http://localhost:8002/health`
2. 防火墙是否开放端口 8002
3. 配置的 URL 是否正确

### Q2: FunASR 服务启动慢？

**A**: 首次启动会下载模型（~3GB），后续启动会快很多。

### Q3: 可以多个主服务共用一个 FunASR 服务吗？

**A**: 可以！这正是独立部署的优势。

### Q4: 如何切换回本地模式？

**A**: 删除 `FUNASR_SERVICE_URL` 配置，重启主服务即可。

---

## ✅ 部署检查清单

### FunASR 服务
- [ ] 虚拟环境已创建
- [ ] 依赖已安装（funasr, torch）
- [ ] `.env` 配置正确
- [ ] 服务启动成功
- [ ] 健康检查通过
- [ ] 测试识别成功

### 主服务
- [ ] `.env` 配置 `FUNASR_SERVICE_URL`
- [ ] **不需要**安装 funasr 依赖
- [ ] 服务启动成功
- [ ] 日志显示 "HTTP模式"
- [ ] 测试音频识别成功

---

**现在你的主服务不需要安装 FunASR 了！** 🎉

参考文档：
- [funasr_standalone/README.md](funasr_standalone/README.md) - FunASR 服务详细说明
- [快速部署指南.md](快速部署指南.md) - 主服务部署
