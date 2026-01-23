# FunASR 本地部署指南

FunASR 是阿里开源的语音识别框架，支持本地部署，无需调用云端API，降低成本。

## 📋 系统要求

### 最低配置
- CPU: 4核以上
- 内存: 8GB以上
- 硬盘: 20GB以上（模型文件需要空间）
- Python: 3.8+

### 推荐配置（GPU加速）
- GPU: NVIDIA GPU（显存4GB+）
- CUDA: 11.7+
- cuDNN: 8.5+

---

## 🚀 安装 FunASR

### 方法1：使用 pip 安装（推荐）

```bash
# 安装 FunASR
pip install funasr modelscope

# 如果有GPU，安装torch（GPU版本）
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 如果只用CPU，安装CPU版本
pip install torch torchaudio
```

### 方法2：从源码安装

```bash
git clone https://github.com/alibaba-damo-academy/FunASR.git
cd FunASR
pip install -e .
```

---

## 📦 下载模型

FunASR 首次运行时会自动下载模型，但为了加快速度，可以手动预下载。

### 方法1：自动下载（推荐）

首次调用时，FunASR 会自动从 ModelScope 下载模型：

```python
from funasr import AutoModel

# 首次运行会自动下载模型（约1-2GB）
model = AutoModel(
    model="paraformer-zh",  # 中文识别模型
    model_revision="v2.0.4"
)
```

**模型下载路径**（默认）：
- Linux/Mac: `~/.cache/modelscope/`
- Windows: `C:\Users\你的用户名\.cache\modelscope\`

### 方法2：手动下载（离线环境）

如果服务器无法访问外网，可以在本地下载后上传：

```bash
# 使用 modelscope cli 下载
pip install modelscope

# 下载模型
modelscope download --model damo/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-pytorch
```

---

## ⚙️ 配置项说明

在 `.env` 文件中配置：

```bash
# ASR服务类型（切换到FunASR）
ASR_SERVICE_TYPE=funasr

# 模型名称（支持多种模型）
FUNASR_MODEL_NAME=paraformer-zh
FUNASR_MODEL_REVISION=v2.0.4

# 运行设备
FUNASR_DEVICE=cpu          # cpu / cuda:0 / cuda:1
FUNASR_NCPU=4              # CPU线程数

# 批处理大小（秒）
FUNASR_BATCH_SIZE=300

# 热词（提升特定词汇识别准确率，用空格分隔）
FUNASR_HOTWORDS=会议纪要 AI 深度学习

# 是否启用VAD（语音活动检测）
FUNASR_ENABLE_VAD=False

# 是否启用标点恢复
FUNASR_ENABLE_PUNC=True
```

---

## 🔄 切换 ASR 服务

### 从腾讯云切换到 FunASR

1. **安装依赖**
```bash
pip install funasr modelscope
```

2. **修改 .env 配置**
```bash
# 将 ASR 服务类型改为 funasr
ASR_SERVICE_TYPE=funasr

# 配置 FunASR 参数
FUNASR_DEVICE=cpu
FUNASR_ENABLE_PUNC=True
```

3. **重启服务**
```bash
python main.py
```

### 从 FunASR 切换回腾讯云

```bash
# 修改 .env
ASR_SERVICE_TYPE=tencent

# 重启服务
python main.py
```

---

## 📊 模型选择

FunASR 支持多种模型，根据需求选择：

| 模型名称 | 语言 | 特点 | 推荐场景 |
|---------|------|------|---------|
| `paraformer-zh` | 中文 | 准确率高，速度快 | **通用场景（推荐）** |
| `paraformer-zh-streaming` | 中文 | 实时流式识别 | 实时会议转写 |
| `paraformer-en` | 英文 | 英文识别 | 英文会议 |
| `conformer` | 中文 | 高精度 | 对准确率要求极高的场景 |

**切换模型**：
```bash
# 修改 .env
FUNASR_MODEL_NAME=paraformer-zh-streaming
```

---

## 🧪 测试 FunASR

### 测试脚本

创建 `test_funasr.py`：

```python
"""
测试 FunASR 本地识别
"""
from funasr import AutoModel

# 初始化模型
print("正在加载模型...")
model = AutoModel(
    model="paraformer-zh",
    model_revision="v2.0.4",
    device="cpu"
)
print("模型加载完成！")

# 测试识别（替换为你的音频文件路径）
audio_file = "test.wav"  # 支持 wav, mp3, m4a, flac 等格式

print(f"开始识别: {audio_file}")
result = model.generate(input=audio_file)

print("\n识别结果:")
print(result)
```

运行：
```bash
python test_funasr.py
```

### 通过API测试

```bash
# 确保服务启动并配置为 FunASR
$env:ASR_SERVICE_TYPE="funasr"
python main.py

# 测试（使用本地音频服务器）
# 终端1: 启动音频服务器
cd D:\test_audio
python -m http.server 9000

# 终端2: 测试API（FunASR支持本地路径）
curl -X POST "http://localhost:8001/api/v1/process" \
  -F "file=@test.wav" \
  -F "template_id=default"
```

---

## 🎯 FunASR vs 腾讯云 ASR

| 对比项 | FunASR 本地 | 腾讯云 ASR |
|-------|------------|-----------|
| **成本** | ✅ 免费（仅硬件成本） | ❌ 按量计费 |
| **部署** | ⚠️ 需要自行部署 | ✅ 无需部署 |
| **性能** | ⚠️ 取决于硬件 | ✅ 稳定高性能 |
| **准确率** | ✅ 高（中文） | ✅ 高 |
| **说话人分离** | ❌ 默认不支持 | ✅ 支持 |
| **实时性** | ✅ 本地处理快 | ⚠️ 依赖网络 |
| **音频限制** | ✅ 支持本地文件 | ❌ 需要公网URL |
| **数据安全** | ✅ 数据不出本地 | ⚠️ 上传到云端 |

**建议**：
- **开发/测试环境**：使用 FunASR（节省成本）
- **生产环境**：根据实际情况选择
  - 对成本敏感：FunASR
  - 对准确率和稳定性要求高：腾讯云 ASR

---

## 🔧 性能优化

### 1. 使用 GPU 加速

```bash
# .env 配置
FUNASR_DEVICE=cuda:0

# 确保安装了 GPU 版本的 PyTorch
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**性能提升**：GPU 比 CPU 快 5-10 倍

### 2. 调整批处理大小

```bash
# 增大批处理，提升吞吐量（需要更多内存）
FUNASR_BATCH_SIZE=600

# 减小批处理，降低内存占用
FUNASR_BATCH_SIZE=150
```

### 3. 使用热词

```bash
# 提升特定词汇识别准确率
FUNASR_HOTWORDS=项目名称 人名 技术术语
```

### 4. 启用标点恢复

```bash
# 自动添加标点符号
FUNASR_ENABLE_PUNC=True
```

---

## 🐛 常见问题

### Q1: 模型下载失败？

**A**: 检查网络，或使用代理：
```bash
export HF_ENDPOINT=https://hf-mirror.com
export MODELSCOPE_CACHE=/path/to/your/cache
```

### Q2: GPU 不可用？

**A**: 检查 CUDA 安装：
```bash
# 检查 CUDA
nvidia-smi

# 检查 PyTorch GPU 支持
python -c "import torch; print(torch.cuda.is_available())"
```

### Q3: 内存不足？

**A**: 减小批处理大小：
```bash
FUNASR_BATCH_SIZE=150
FUNASR_DEVICE=cpu
FUNASR_NCPU=2
```

### Q4: 识别准确率低？

**A**: 尝试以下方法：
1. 使用热词提升特定词汇准确率
2. 确保音频质量良好（16kHz采样率）
3. 尝试更大的模型（如 conformer）

---

## 📚 参考资料

- **FunASR 官方文档**: https://github.com/alibaba-damo-academy/FunASR
- **ModelScope 模型库**: https://modelscope.cn/models
- **FunASR 模型列表**: https://github.com/alibaba-damo-academy/FunASR/blob/main/docs/model_zoo/README.md

---

祝部署顺利！🎉
