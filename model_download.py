import os
from huggingface_hub import snapshot_download

# 1. 设置你的 Token (如果在环境变量里设置了，这里可以省略)
HF_TOKEN = " "

# 2. 指定下载目录 (基于你的截图，就是当前目录下的 model 文件夹)
# 这里的 "." 表示当前脚本所在目录
download_path = os.path.join(os.getcwd(), "models")

print(f"准备将模型下载到: {download_path}")

# 3. 下载 Pyannote 的主模型 (Diarization)
# local_dir_use_symlinks=False 这一点很重要，确保下载的是真实文件而不是快捷方式
print("正在下载 Speaker Diarization 模型...")
path1 = snapshot_download(
    repo_id="pyannote/speaker-diarization-3.1",
    local_dir=os.path.join(download_path, "pyannote_diarization"),
    token=HF_TOKEN,
    local_dir_use_symlinks=False 
)

# 4. 下载依赖的 Segmentation 模型 (必须下载，否则运行会报错)
print("正在下载 Segmentation 模型...")
path2 = snapshot_download(
    repo_id="pyannote/segmentation-3.0",
    local_dir=os.path.join(download_path, "pyannote_segmentation"),
    token=HF_TOKEN,
    local_dir_use_symlinks=False
)

print("\n下载完成！")
print(f"主模型位置: {path1}")
print(f"分割模型位置: {path2}")