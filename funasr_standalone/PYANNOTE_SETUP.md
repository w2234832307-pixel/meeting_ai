# Pyannote 说话人分离设置指南

## 简介

Pyannote.audio 是目前最先进的开源说话人分离模型，准确率远高于传统的聚类方法。

## 安装步骤

### 1. 安装 Pyannote.audio

```bash
pip install pyannote.audio
```

### 2. 在 HuggingFace 上接受模型使用协议

1. 访问：https://huggingface.co/pyannote/speaker-diarization-3.1
2. 登录你的 HuggingFace 账号（如果没有，需要注册）
3. 点击 "Accept" 接受模型使用协议
4. 如果需要，生成一个 Access Token：
   - 访问：https://huggingface.co/settings/tokens
   - 创建新 token（read 权限即可）

### 3. 配置环境变量（可选）

如果需要使用 token（访问私有模型或避免重复授权）：

```bash
# Windows
set HF_TOKEN=your_huggingface_token_here

# Linux/Mac
export HF_TOKEN=your_huggingface_token_here
```

或者在代码中直接使用（不推荐，安全性差）。

### 4. 启用 Pyannote

设置环境变量启用 Pyannote：

```bash
# Windows
set USE_PYANNOTE=true

# Linux/Mac
export USE_PYANNOTE=true
```

或者在启动服务时：

```bash
USE_PYANNOTE=true python -m uvicorn main:app --host 0.0.0.0 --port 8000
```

## 使用说明

### 自动降级

- 如果 Pyannote 未安装或加载失败，系统会自动降级使用 Cam++ 方案
- 不会影响服务的正常运行

### 性能对比

| 方案 | 准确率 | 速度 | 资源占用 |
|------|--------|------|----------|
| Cam++ 聚类 | 60-70% | 快 | 低 |
| Pyannote | 85-95% | 中等 | 中等 |

### 注意事项

1. **首次使用**：首次运行会下载模型（约 1-2GB），需要一些时间
2. **GPU 支持**：Pyannote 支持 GPU 加速，会自动使用 CUDA（如果可用）
3. **内存占用**：Pyannote 需要更多内存，建议至少 8GB RAM

## 故障排除

### 问题1：ImportError: No module named 'pyannote'

**解决**：安装 Pyannote
```bash
pip install pyannote.audio
```

### 问题2：OSError: Can't load tokenizer

**解决**：需要在 HuggingFace 上接受模型使用协议
1. 访问：https://huggingface.co/pyannote/speaker-diarization-3.1
2. 登录并接受协议

### 问题3：模型下载失败

**解决**：
1. 检查网络连接
2. 如果在中国，可能需要配置代理
3. 或者手动下载模型到本地

### 问题4：内存不足

**解决**：
1. 关闭其他占用内存的程序
2. 使用 CPU 版本（虽然慢一些）
3. 处理较短的音频文件

## 代码示例

### 在代码中使用

```python
import os

# 启用 Pyannote
os.environ["USE_PYANNOTE"] = "true"

# 如果需要 token
os.environ["HF_TOKEN"] = "your_token_here"
```

### 直接调用 Pyannote 函数

```python
from pyannote_diarization import perform_pyannote_diarization

transcript = [
    {"text": "你好", "start_time": 0.0, "end_time": 1.0},
    {"text": "世界", "start_time": 1.0, "end_time": 2.0},
]

result = perform_pyannote_diarization(
    audio_path="audio.wav",
    transcript=transcript,
    use_auth_token="your_token_here"  # 可选
)
```

## 更多信息

- Pyannote 官方文档：https://github.com/pyannote/pyannote-audio
- HuggingFace 模型页面：https://huggingface.co/pyannote/speaker-diarization-3.1
