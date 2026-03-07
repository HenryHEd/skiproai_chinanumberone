# 将 python:3.9-slim 改为 python:3.10-slim 或更高
FROM python:3.11-slim

WORKDIR /app

# 安装核心依赖
RUN apt-get update --fix-missing && apt-get install -y \
    build-essential \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 复制项目文件
COPY . .

# 升级 pip 并安装依赖
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# 暴露端口
EXPOSE 8501

# 启动
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
