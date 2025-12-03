# 使用官方Python基础镜像，支持 CUDA 12.6（与你本地环境一致）
FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（带简单重试，避免源 502/网络波动导致构建失败）
RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
        python3.10 \
        python3-pip \
        python3-dev \
        git \
        wget \
        curl \
        build-essential \
        libgl1-mesa-glx \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender-dev \
        libgomp1 \
        ffmpeg \
        libegl1 \
        libegl1-mesa \
        libgles2 \
        libgles2-mesa \
        libgl1-mesa-dev \
    || ( \
        echo '第一次 apt 安装失败，重试一次...'; \
        sleep 20; \
        apt-get update; \
        apt-get install -y --no-install-recommends \
            python3.10 \
            python3-pip \
            python3-dev \
            git \
            wget \
            curl \
            build-essential \
            libgl1-mesa-glx \
            libglib2.0-0 \
            libsm6 \
            libxext6 \
            libxrender-dev \
            libgomp1 \
            ffmpeg \
            libegl1 \
            libegl1-mesa \
            libgles2 \
            libgles2-mesa \
            libgl1-mesa-dev \
    ); \
    rm -rf /var/lib/apt/lists/*

# 安装 Node.js 20（Vite 5 要求 Node >= 18）
RUN set -eux; \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -; \
    apt-get update; \
    apt-get install -y --no-install-recommends nodejs; \
    rm -rf /var/lib/apt/lists/*

# 创建符号链接
RUN ln -s /usr/bin/python3 /usr/bin/python

# 升级pip
RUN pip3 install --upgrade pip setuptools wheel

# 复制requirements文件
COPY requirements.txt .

# 先安装 PyTorch (CUDA 12.6 版本)，因为 detectron2 需要它
# 使用 cu126 对应的官方轮子
RUN pip3 install \
    torch==2.6.0+cu126 \
    torchvision==0.21.0+cu126 \
    torchaudio==2.6.0+cu126 \
    --extra-index-url https://download.pytorch.org/whl/cu126

# 安装detectron2 (需要从源码构建，必须在PyTorch之后)
# 使用 --no-build-isolation 让其在已有 torch 环境中构建，避免构建环境内找不到 torch
RUN pip3 install --no-build-isolation 'git+https://github.com/facebookresearch/detectron2.git'

# 安装Python依赖
RUN pip3 install -r requirements.txt

# 安装 MoGe (FOV Estimator)
RUN pip3 install 'git+https://github.com/microsoft/MoGe.git'

# 复制项目文件
COPY . .

# 将本机已下载的模型缓存一起打包进镜像
# HuggingFace 缓存（SAM-3D-Body、MoGe2 等）
COPY local_models/huggingface /root/.cache/huggingface
# torch 缓存（dinov3 等）
COPY local_models/torch_cache /root/.cache/torch
# torch hub / detectron2 权重等
COPY local_models/torch_home /root/.torch

# 构建前端
WORKDIR /app/frontend
RUN if [ -f package.json ]; then \
        npm install && \
        npm run build; \
    fi

# 回到工作目录
WORKDIR /app

# 创建必要的目录
RUN mkdir -p uploads outputs data

# 暴露端口
EXPOSE 5000

# 设置启动脚本（先复制，然后转换行尾符并设置权限）
# 注意：Windows 文件可能有 CRLF 行尾符，需要转换为 LF
COPY docker-entrypoint.sh /tmp/docker-entrypoint.sh
RUN sed -i 's/\r$//' /tmp/docker-entrypoint.sh && \
    mv /tmp/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh && \
    chmod +x /usr/local/bin/docker-entrypoint.sh

# 启动命令
ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["python3", "app.py"]

