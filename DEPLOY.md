# Docker 部署指南

## 前置要求

1. **服务器环境**：
   - Ubuntu 20.04+ 或 CentOS 7+
   - Docker 20.10+
   - Docker Compose 2.0+
   - NVIDIA Docker Runtime（nvidia-container-toolkit）
   - 7张GPU（GPU 0-6）

2. **验证GPU支持**：
   ```bash
   # 检查NVIDIA驱动
   nvidia-smi
   
   # 检查Docker GPU支持
   docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
   ```

## 快速部署步骤

### 1. 上传文件到服务器

将整个项目目录上传到服务器，例如：
```bash
scp -r meeting_ai/ user@server:/opt/meeting_ai/
```

### 2. 配置环境变量

```bash
cd /opt/meeting_ai
cp .env.example .env
# 编辑 .env 文件，填写实际配置
nano .env
```

**必须配置的项**：
- `LLM_API_KEY`：LLM API密钥
- `CHROMA_HOST`：Chroma向量数据库地址
- `CHROMA_PORT`：Chroma端口
- 其他根据实际情况配置

### 3. 构建并启动服务

```bash
# 构建镜像并启动所有服务
docker compose up -d --build

# 查看服务状态
docker compose ps

# 查看日志
docker compose logs -f
```

### 4. 验证服务

```bash
# 检查主服务
curl http://localhost:8000/health

# 检查FunASR服务（GPU 0）
curl http://localhost:8002/health

# 检查Pyannote服务（GPU 0）
curl http://localhost:8100/health
```

## 服务说明

### 服务列表

- **主服务** (meeting-ai): 端口 8000
- **FunASR服务** (7个实例):
  - GPU 0: 端口 8002
  - GPU 1: 端口 8003
  - GPU 2: 端口 8004
  - GPU 3: 端口 8005
  - GPU 4: 端口 8006
  - GPU 5: 端口 8007
  - GPU 6: 端口 8008
- **Pyannote服务** (7个实例):
  - GPU 0: 端口 8100
  - GPU 1: 端口 8101
  - GPU 2: 端口 8102
  - GPU 3: 端口 8103
  - GPU 4: 端口 8104
  - GPU 5: 端口 8105
  - GPU 6: 端口 8106

### 常用命令

```bash
# 启动所有服务
docker compose up -d

# 停止所有服务
docker compose down

# 重启所有服务
docker compose restart

# 查看特定服务日志
docker compose logs -f meeting-ai
docker compose logs -f funasr-gpu0

# 进入容器
docker exec -it meeting-ai bash

# 查看GPU使用情况
nvidia-smi
```

## 故障排查

### 1. 容器启动失败

```bash
# 查看详细错误
docker compose logs [service_name]

# 检查GPU是否可用
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### 2. 端口冲突

如果端口被占用，修改 `docker-compose.yml` 中的端口映射：
```yaml
ports:
  - "新端口:容器端口"
```

### 3. GPU不可用

确保已安装 nvidia-container-toolkit：
```bash
# Ubuntu
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### 4. 内存不足

如果构建镜像时内存不足，可以：
- 增加服务器swap空间
- 分批构建服务（先构建主服务，再构建其他服务）

## 性能优化

1. **GPU显存**：每张GPU建议至少11GB显存
2. **并发数**：根据显存大小调整每张GPU的并发数
3. **监控**：建议使用 `nvidia-smi` 或 Prometheus 监控GPU使用率

## 注意事项

1. **首次启动**：FunASR和Pyannote服务首次启动需要下载模型，可能需要较长时间
2. **网络**：确保容器之间可以互相访问（使用Docker网络）
3. **存储**：确保有足够的磁盘空间存储模型和临时文件
4. **日志**：日志文件会保存在 `./logs` 目录，建议定期清理
