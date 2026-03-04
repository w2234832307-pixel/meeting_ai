# pdf_service/main.py
import os
import subprocess
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse

app = FastAPI()

TEMP_DIR = "/tmp/pdf_convert"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.post("/convert", response_class=HTMLResponse)
async def convert_pdf(file: UploadFile = File(...)):
    # 1. 生成唯一文件名，避免冲突
    unique_id = str(uuid.uuid4())
    pdf_filename = f"{unique_id}.pdf"
    html_filename = f"{unique_id}.html"
    
    pdf_path = os.path.join(TEMP_DIR, pdf_filename)
    html_path = os.path.join(TEMP_DIR, html_filename)

    try:
        # 2. 保存上传的 PDF
        with open(pdf_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)

        # 3. 调用本地命令 pdf2htmlEX (因为我们在容器里，它就是本地命令)
        # --zoom 1.3 放大一点看起来更清晰
        cmd = [
            "pdf2htmlEX",
            "--zoom", "1.3",
            "--embed", "cfijo",  # 嵌入所有资源
            "--dest-dir", TEMP_DIR,
            pdf_path,
            html_filename # 指定输出文件名
        ]
        
        # 执行转换
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # 4. 读取生成的 HTML
        if os.path.exists(html_path):
            with open(html_path, "r", encoding="utf-8") as f:
                html_content = f.read()
            return html_content
        else:
            raise HTTPException(status_code=500, detail="Conversion failed: Output file not found")

    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Conversion error: {e.stderr.decode()}")
    finally:
        # 5. 清理临时文件 (非常重要，防止容器爆盘)
        if os.path.exists(pdf_path): os.remove(pdf_path)
        if os.path.exists(html_path): os.remove(html_path)

@app.get("/health")
def health_check():
    return {"status": "ok"}