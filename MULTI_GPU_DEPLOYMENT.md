# 多GPU多实例部署方案

## 概述

当服务器有多张GPU（如7张）时，可以通过**多实例部署**的方式充分利用所有GPU，实现真正的并行处理。

## 方案说明

### 核心思路

1. **FunASR服务**：为每张GPU启动一个独立实例，每个实例绑定不同的GPU
2. **Pyannote服务**：为每张GPU启动一个独立实例，每个实例绑定不同的GPU
3. **主服务**：根据GPU ID自动选择对应的服务实例URL

### 部署架构

```
主服务 (meeting_ai)
  ├─ GPU池管理器 (自动检测7张GPU)
  │
  ├─ FunASR服务实例
  │   ├─ 实例0 (端口8002, GPU 0)
  │   ├─ 实例1 (端口8003, GPU 1)
  │   ├─ 实例2 (端口8004, GPU 2)
  │   ├─ ...
  │   └─ 实例6 (端口8008, GPU 6)
  │
  └─ Pyannote服务实例
      ├─ 实例0 (端口8100, GPU 0)
      ├─ 实例1 (端口8101, GPU 1)
      ├─ 实例2 (端口8102, GPU 2)
      ├─ ...
      └─ 实例6 (端口8106, GPU 6)
```

## Docker部署方案

### 方案1：使用docker-compose（推荐）

创建 `docker-compose.multi-gpu.yml`：

```yaml
version: '3.8'

services:
  # FunASR服务 - GPU 0
  funasr-gpu0:
    image: your-funasr-image:latest
    container_name: funasr-gpu0
    ports:
      - "8002:8002"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    command: python funasr_standalone/main.py --port 8002

  # FunASR服务 - GPU 1
  funasr-gpu1:
    image: your-funasr-image:latest
    container_name: funasr-gpu1
    ports:
      - "8003:8002"
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    command: python funasr_standalone/main.py --port 8002

  # ... 依此类推，创建7个FunASR实例

  # Pyannote服务 - GPU 0
  pyannote-gpu0:
    image: your-pyannote-image:latest
    container_name: pyannote-gpu0
    ports:
      - "8100:8100"
    environment:
      - CUDA_VISIBLE_DEVICES=0
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['0']
              capabilities: [gpu]
    command: python funasr_standalone/pyannote_server.py --port 8100

  # Pyannote服务 - GPU 1
  pyannote-gpu1:
    image: your-pyannote-image:latest
    container_name: pyannote-gpu1
    ports:
      - "8101:8100"
    environment:
      - CUDA_VISIBLE_DEVICES=1
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              device_ids: ['1']
              capabilities: [gpu]
    command: python funasr_standalone/pyannote_server.py --port 8100

  # ... 依此类推，创建7个Pyannote实例

  # 主服务
  meeting-ai:
    image: your-meeting-ai-image:latest
    container_name: meeting-ai
    ports:
      - "8000:8000"
    environment:
      - FUNASR_SERVICE_URL=http://funasr-gpu0:8002  # 默认URL，实际会动态选择
      - PYANNOTE_SERVICE_URL=http://pyannote-gpu0:8100  # 默认URL，实际会动态选择
    depends_on:
      - funasr-gpu0
      - funasr-gpu1
      # ... 其他实例
      - pyannote-gpu0
      - pyannote-gpu1
      # ... 其他实例
```

### 方案2：手动启动多个容器

```bash
# FunASR服务 - 7个实例
docker run -d --name funasr-gpu0 --gpus device=0 -p 8002:8002 \
  -e CUDA_VISIBLE_DEVICES=0 \
  your-funasr-image:latest python funasr_standalone/main.py --port 8002

docker run -d --name funasr-gpu1 --gpus device=1 -p 8003:8002 \
  -e CUDA_VISIBLE_DEVICES=1 \
  your-funasr-image:latest python funasr_standalone/main.py --port 8002

# ... 依此类推

docker run -d --name funasr-gpu6 --gpus device=6 -p 8008:8002 \
  -e CUDA_VISIBLE_DEVICES=6 \
  your-funasr-image:latest python funasr_standalone/main.py --port 8002

# Pyannote服务 - 7个实例
docker run -d --name pyannote-gpu0 --gpus device=0 -p 8100:8100 \
  -e CUDA_VISIBLE_DEVICES=0 \
  your-pyannote-image:latest python funasr_standalone/pyannote_server.py --port 8100

docker run -d --name pyannote-gpu1 --gpus device=1 -p 8101:8100 \
  -e CUDA_VISIBLE_DEVICES=1 \
  your-pyannote-image:latest python funasr_standalone/pyannote_server.py --port 8100

# ... 依此类推

docker run -d --name pyannote-gpu6 --gpus device=6 -p 8106:8100 \
  -e CUDA_VISIBLE_DEVICES=6 \
  your-pyannote-image:latest python funasr_standalone/pyannote_server.py --port 8100
```

## 主服务改造

需要修改主服务，使其能够根据GPU ID动态选择对应的服务URL。

### 修改 `app/api/endpoints.py`

在 `handle_audio_parallel` 函数中，根据 `gpu_device` 参数选择对应的服务URL：

```python
async def handle_audio_parallel(audio_path: str, is_url: bool, asr_model: str, gpu_device: Optional[str] = None):
    """
    封装并行处理逻辑 (全异步流式 I/O)
    """
    from app.services.parallel_processor import map_words_to_speakers, aggregate_by_speaker, parse_rttm
    from app.services.gpu_pool import get_gpu_pool
    
    # 根据GPU设备选择对应的服务URL
    base_funasr_url = os.getenv("FUNASR_SERVICE_URL", "http://localhost:8002")
    base_pyannote_url = os.getenv("PYANNOTE_SERVICE_URL", "http://localhost:8100")
    
    if gpu_device and gpu_device.startswith("cuda:"):
        try:
            gpu_id = int(gpu_device.split(":")[1])
            # FunASR服务端口：8002 + gpu_id
            funasr_url = f"http://funasr-gpu{gpu_id}:8002" if gpu_id < 7 else base_funasr_url
            # Pyannote服务端口：8100 + gpu_id
            pyannote_url = f"http://pyannote-gpu{gpu_id}:8100" if gpu_id < 7 else base_pyannote_url
        except (ValueError, IndexError):
            funasr_url = base_funasr_url
            pyannote_url = base_pyannote_url
    else:
        funasr_url = base_funasr_url
        pyannote_url = base_pyannote_url
    
    # ... 其余代码保持不变
```

## 非Docker部署方案

如果使用非Docker部署，可以通过环境变量和启动参数实现：

### 启动脚本示例

创建 `start_multi_gpu.sh`：

```bash
#!/bin/bash

# FunASR服务 - 7个实例
for i in {0..6}; do
    CUDA_VISIBLE_DEVICES=$i python funasr_standalone/main.py --port $((8002 + $i)) &
    echo "启动 FunASR 实例 $i (GPU $i, 端口 $((8002 + $i)))"
done

# Pyannote服务 - 7个实例
for i in {0..6}; do
    CUDA_VISIBLE_DEVICES=$i python funasr_standalone/pyannote_server.py --port $((8100 + $i)) &
    echo "启动 Pyannote 实例 $i (GPU $i, 端口 $((8100 + $i)))"
done

# 等待所有服务启动
sleep 10

# 启动主服务
python main.py
```

## 配置说明

### 环境变量

主服务需要配置服务发现机制，可以通过以下方式：

1. **固定端口映射**（推荐）：
   - FunASR: GPU 0 → 8002, GPU 1 → 8003, ..., GPU 6 → 8008
   - Pyannote: GPU 0 → 8100, GPU 1 → 8101, ..., GPU 6 → 8106

2. **服务发现**（如果使用K8s等）：
   - 使用Service名称：`funasr-gpu0`, `funasr-gpu1`, ...
   - 使用Service名称：`pyannote-gpu0`, `pyannote-gpu1`, ...

### 主服务配置

在 `.env` 或环境变量中：

```bash
# 基础URL（用于服务发现）
FUNASR_SERVICE_URL=http://funasr-gpu0:8002
PYANNOTE_SERVICE_URL=http://pyannote-gpu0:8100

# 或者使用负载均衡器（如果配置了）
# FUNASR_SERVICE_URL=http://funasr-lb:8002
# PYANNOTE_SERVICE_URL=http://pyannote-lb:8100
```

## 验证部署

### 1. 检查服务状态

```bash
# 检查FunASR服务
curl http://localhost:8002/health
curl http://localhost:8003/health
# ... 检查所有实例

# 检查Pyannote服务
curl http://localhost:8100/health
curl http://localhost:8101/health
# ... 检查所有实例
```

### 2. 测试GPU分配

发送多个并发请求，观察日志：

```bash
# 同时发送7个请求
for i in {1..7}; do
    curl -X POST http://localhost:8000/api/v1/process \
      -F "audio_urls=[\"http://example.com/audio$i.mp3\"]" &
done
```

应该看到日志中显示不同的GPU被分配：
```
🎯 使用GPU处理音频: cuda:0
🎯 使用GPU处理音频: cuda:1
🎯 使用GPU处理音频: cuda:2
...
🎯 使用GPU处理音频: cuda:6
```

## 性能优化建议

1. **每张GPU最大并发数**：根据显存大小调整
   - 11GB显存：建议每张GPU最大并发1-2
   - 24GB显存：建议每张GPU最大并发2-3

2. **负载均衡**：如果请求不均匀，可以考虑使用Nginx等负载均衡器

3. **监控**：建议添加GPU使用率监控，确保所有GPU都被充分利用

## 注意事项

1. **端口管理**：确保端口不冲突，建议使用端口范围：
   - FunASR: 8002-8008
   - Pyannote: 8100-8106

2. **资源限制**：每个容器/进程都会占用内存和CPU，确保服务器资源充足

3. **网络配置**：如果使用Docker，确保容器之间可以互相访问（使用docker网络或host网络）

4. **故障恢复**：建议使用进程管理工具（如supervisor、systemd）或容器编排工具（如K8s）来管理服务生命周期
