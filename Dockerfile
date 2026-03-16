# 使用支持CUDA 12.1的Python镜像
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

# 安装系统依赖（增加软件源提速，可选）
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list && \
    apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsndfile1 \
    libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建符号链接（使用 -sf 强制覆盖已有的链接）
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/pip3 /usr/bin/pip

# 设置工作目录
WORKDIR /app

# ================= 核心防御策略：建立“存档点” =================

# 1. 全局配置：使用清华源，并将超时时间从默认的 15 秒拉长到 1000 秒，防止小包断流
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple && \
    pip config set global.timeout 1000

# 2. 升级 pip 自身，解决旧版本解析依赖树容易崩溃的问题
RUN pip install --upgrade pip

# 3. 【绝对核心】单独下载并缓存最大的“毒瘤”包（Torch全家桶）
# 只要这一步跑完，哪怕后面服务器断电，这 5GB 都会永久保存在硬盘的 Docker 缓存层里，永不重下！
RUN pip install torch==2.8.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121 \
    --extra-index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 4. 安装 FunASR 和算法依赖（去掉了 --no-cache-dir，就算断网重试也能断点续传）
COPY funasr_standalone/requirements.txt ./funasr_standalone/requirements.txt
RUN pip install -r funasr_standalone/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu121

RUN pip install addict

# 5. 安装主程序的依赖
COPY requirements.txt .
RUN pip install -r requirements.txt

# ==========================================================

# 复制所有代码
COPY . .

# 创建必要的目录
RUN mkdir -p logs temp_files

# 暴露端口
EXPOSE 8000 8002 8100

# 默认启动命令
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]