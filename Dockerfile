# petread —— Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

# 系统依赖:
#   ffmpeg          —— 视频解码 / 抽音轨
#   libgl1          —— opencv 的 OpenGL 运行时依赖
#   libglib2.0-0    —— opencv 的 glib/gthread 运行时依赖(headless 版仍需要)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖,利用 Docker 层缓存:requirements 不变时不必重装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 再复制项目文件(.dockerignore 已排除 .venv / data / .git 等)
COPY . .

# HF Spaces 固定路由到 7860;app.py 读 PORT,默认也是 7860。host 由代码固定 0.0.0.0。
ENV PORT=7860
# 数据目录默认 /app/data(容器内可写)。HF 上如开了持久化存储可在 Space 设 DATA_DIR=/data。
ENV DATA_DIR=/app/data
EXPOSE 7860

CMD ["python", "app.py"]
