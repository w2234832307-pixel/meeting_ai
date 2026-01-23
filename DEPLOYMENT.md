# 部署指南（生产环境）

## 📋 部署方式说明

### 开发环境 vs 生产环境

| 环境 | 访问地址 | 说明 |
|------|----------|------|
| **开发环境**（本地） | `http://localhost:8001` | 只能在你的电脑上访问 |
| **生产环境**（服务器） | `http://your-server-ip:8001` 或 `https://api.yourdomain.com` | 可以被其他人访问 |

**重要**：`localhost` 只能在本地访问，如果要给前端同事使用，必须部署到服务器！

---

## 🚀 部署方式

### 方式1：Docker 部署（推荐）

Docker 可以确保环境一致性，是生产环境的标准做法。

#### 步骤1：创建 Dockerfile

```dockerfile
FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装 Python 依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目代码
COPY . .

# 创建日志和临时文件目录
RUN mkdir -p logs temp_files

# 暴露端口
EXPOSE 8001

# 启动命令
CMD ["python", "main.py"]
```

#### 步骤2：创建 .dockerignore

```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.env
.git
.gitignore
logs/
temp_files/
*.log
test_*.py
*.md
```

#### 步骤3：构建镜像

```bash
docker build -t meeting-ai:latest .
```

#### 步骤4：运行容器

```bash
docker run -d \
  --name meeting-ai \
  -p 8001:8001 \
  -v $(pwd)/logs:/app/logs \
  -v $(pwd)/temp_files:/app/temp_files \
  --env-file .env \
  meeting-ai:latest
```

**说明**：
- `-d`：后台运行
- `-p 8001:8001`：映射端口
- `-v`：挂载日志和临时文件目录（持久化）
- `--env-file .env`：加载环境变量

#### 步骤5：查看日志

```bash
docker logs -f meeting-ai
```

#### 步骤6：停止/重启

```bash
# 停止
docker stop meeting-ai

# 重启
docker restart meeting-ai

# 删除容器
docker rm -f meeting-ai
```

---

### 方式2：直接在服务器部署

#### 步骤1：上传代码到服务器

```bash
# 使用 git
git clone your-repo-url
cd meeting_ai

# 或使用 scp
scp -r meeting_ai/ user@server:/path/to/meeting_ai
```

#### 步骤2：安装依赖

```bash
# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

#### 步骤3：配置环境变量

```bash
# 复制配置文件
cp env.example .env

# 编辑配置
vim .env
```

#### 步骤4：使用 Gunicorn 运行（生产环境推荐）

```bash
# 安装 gunicorn
pip install gunicorn

# 启动服务
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --timeout 300 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  --daemon
```

**参数说明**：
- `--workers 4`：4个工作进程（根据CPU核心数调整）
- `--worker-class uvicorn.workers.UvicornWorker`：使用 Uvicorn worker
- `--bind 0.0.0.0:8001`：监听所有网络接口的8001端口
- `--timeout 300`：超时时间5分钟
- `--daemon`：后台运行

#### 步骤5：使用 systemd 管理服务（推荐）

创建 `/etc/systemd/system/meeting-ai.service`：

```ini
[Unit]
Description=Meeting AI Service
After=network.target

[Service]
Type=forking
User=your-user
Group=your-group
WorkingDirectory=/path/to/meeting_ai
Environment="PATH=/path/to/meeting_ai/venv/bin"
ExecStart=/path/to/meeting_ai/venv/bin/gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8001 \
  --timeout 300 \
  --access-logfile /path/to/meeting_ai/logs/access.log \
  --error-logfile /path/to/meeting_ai/logs/error.log \
  --daemon
ExecReload=/bin/kill -s HUP $MAINPID
ExecStop=/bin/kill -s TERM $MAINPID
PrivateTmp=true

[Install]
WantedBy=multi-user.target
```

启动服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable meeting-ai
sudo systemctl start meeting-ai
sudo systemctl status meeting-ai
```

---

### 方式3：使用 Nginx 反向代理（推荐配合使用）

Nginx 可以提供：
- HTTPS 支持
- 负载均衡
- 静态文件服务
- 请求限流

#### Nginx 配置示例

创建 `/etc/nginx/sites-available/meeting-ai`：

```nginx
server {
    listen 80;
    server_name api.yourdomain.com;  # 改成你的域名

    # 请求体大小限制（上传大文件）
    client_max_body_size 500M;

    # 代理到 Python 后端
    location / {
        proxy_pass http://localhost:8001;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # 超时设置（ASR处理需要较长时间）
        proxy_connect_timeout 300s;
        proxy_send_timeout 300s;
        proxy_read_timeout 300s;
    }
}
```

启用配置：
```bash
sudo ln -s /etc/nginx/sites-available/meeting-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

#### 配置 HTTPS（使用 Let's Encrypt）

```bash
# 安装 certbot
sudo apt install certbot python3-certbot-nginx

# 获取证书
sudo certbot --nginx -d api.yourdomain.com

# 自动续期
sudo systemctl enable certbot.timer
```

---

## 🔗 给前端同事的接口地址

### 开发环境（本地测试）
```
http://localhost:8001
```
**注意**：只能在你自己电脑上访问，前端同事访问不到！

### 生产环境（部署到服务器后）

**方式1：使用服务器IP**
```
http://your-server-ip:8001
```
例如：`http://192.168.1.100:8001`

**方式2：使用域名（推荐）**
```
https://api.yourdomain.com
```
例如：`https://api.meeting.example.com`

**方式3：使用 Nginx 反向代理 + 子路径**
```
https://yourdomain.com/api/meeting-ai
```

---

## 📝 前端集成示例

给前端同事的调用示例：

```javascript
// 配置
const API_BASE_URL = 'https://api.yourdomain.com';  // 生产环境地址

// 处理音频
async function processAudio(audioUrl) {
  const formData = new FormData();
  formData.append('audio_url', audioUrl);
  formData.append('template_id', 'default');
  
  try {
    const response = await fetch(`${API_BASE_URL}/api/v1/process`, {
      method: 'POST',
      body: formData
    });
    
    if (response.ok) {
      const result = await response.json();
      console.log('处理成功:', result);
      
      // 使用逐字稿实现点击跳转
      result.transcript.forEach(item => {
        console.log(`${item.text} (${item.start_time}s - ${item.end_time}s)`);
      });
      
      return result;
    } else {
      console.error('处理失败:', response.status);
    }
  } catch (error) {
    console.error('请求错误:', error);
  }
}

// 音频时间跳转功能
function jumpToAudioTime(audioPlayer, timestamp) {
  audioPlayer.currentTime = timestamp;
  audioPlayer.play();
}

// 使用示例
const audioPlayer = document.getElementById('audio-player');
const transcript = result.transcript;

// 点击文本跳转到对应音频位置
transcript.forEach(item => {
  const textElement = document.createElement('span');
  textElement.textContent = item.text;
  textElement.onclick = () => jumpToAudioTime(audioPlayer, item.start_time);
  document.body.appendChild(textElement);
});
```

---

## 🔒 安全建议

### 1. API 认证（可选）

在生产环境建议添加 API Key 认证：

```python
# app/api/endpoints.py
from fastapi import Header, HTTPException

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != settings.API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")

@router.post("/process", dependencies=[Depends(verify_api_key)])
async def process_meeting_audio(...):
    ...
```

### 2. 限流（Rate Limiting）

使用 Nginx 或 FastAPI 中间件限制请求频率。

### 3. CORS 配置

在 `main.py` 中配置允许的前端域名：

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",  # 生产环境前端域名
        "http://localhost:3000",   # 开发环境
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 📊 监控和日志

### 1. 查看日志
```bash
# 实时查看日志
tail -f logs/app.log

# 查看错误日志
grep ERROR logs/app.log

# 查看访问日志
tail -f logs/access.log
```

### 2. 健康检查
```bash
curl http://localhost:8001/health
```

### 3. 监控指标（可选）
- 集成 Prometheus + Grafana
- 监控 CPU、内存、请求响应时间
- 设置告警规则

---

## ✅ 部署检查清单

部署前确认：

- [ ] `.env` 文件配置完整（密钥、数据库等）
- [ ] 依赖已全部安装（`pip install -r requirements.txt`）
- [ ] 端口未被占用（8001）
- [ ] 防火墙规则已配置（允许8001端口）
- [ ] 日志目录有写权限
- [ ] 临时文件目录有写权限
- [ ] MySQL/Milvus 服务正常运行（如果使用）
- [ ] 音频文件必须使用可公网访问的URL

部署后确认：

- [ ] 服务启动成功
- [ ] 访问 `/docs` 能看到API文档
- [ ] 访问 `/health` 返回正常
- [ ] 测试一次完整的API调用
- [ ] 查看日志无错误
- [ ] 前端能正常调用

---

## 🆘 常见问题

### Q1: 前端同事说访问不到我的服务？
**A**: `localhost` 只能在本地访问。你需要：
1. 部署到服务器
2. 给前端同事服务器的IP或域名
3. 确保服务器防火墙允许8001端口访问

### Q2: 如何在局域网内测试？
**A**: 
```bash
# 查看本机IP
ipconfig  # Windows
ifconfig  # Linux/Mac

# 假设你的IP是 192.168.1.100
# 前端同事可以访问：http://192.168.1.100:8001
```

### Q3: 生产环境需要什么配置？
**A**: 
- 服务器（至少2核4G内存）
- 域名（推荐）
- HTTPS证书（Let's Encrypt免费）
- Nginx反向代理
- Gunicorn 多进程部署

---

祝部署顺利！🚀
