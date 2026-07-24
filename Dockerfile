# petread —— 容器镜像(Railway / 任意支持 Dockerfile 的平台)
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

# 端口不写死:运行平台通过 PORT 环境变量注入(Railway 会自动注入),
# app.py 读 os.environ["PORT"],缺省 7860,host 由代码固定 0.0.0.0。
# EXPOSE 仅作文档提示,不绑定实际端口;用 ARG 给个默认值,不固定 7860。
ARG PORT=7860
EXPOSE ${PORT}

# 数据目录默认 /app/data(容器内可写)。Railway 同为临时文件系统,重启即丢。
ENV DATA_DIR=/app/data

CMD ["python", "app.py"]
