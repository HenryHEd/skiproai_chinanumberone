# 使用轻量级 Python 镜像
FROM python:3.9-slim

WORKDIR /app

# 安装系统依赖（如果你的 app.py 用到了 OpenCV 或其它库）
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制当前目录所有文件到容器
COPY . .

# 安装依赖
RUN pip install --no-cache-dir -r requirements.txt

# 暴露 Streamlit 默认端口
EXPOSE 8501

# 启动命令：强制指定端口和地址
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
