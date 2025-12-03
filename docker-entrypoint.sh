#!/bin/bash
set -e

echo "=========================================="
echo "SAM-3D-Body Docker 容器启动中..."
echo "=========================================="

# 检查CUDA是否可用
if command -v nvidia-smi &> /dev/null; then
    echo "检测到NVIDIA GPU:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "警告: 未检测到NVIDIA GPU，将使用CPU模式"
    echo "注意: CPU模式运行速度较慢"
fi

# 创建必要的目录
mkdir -p /app/uploads
mkdir -p /app/outputs
mkdir -p /app/data

# 设置权限
chmod -R 755 /app/uploads /app/outputs /app/data

# 检查模型是否已下载
if [ ! -d "$HOME/.cache/huggingface" ] || [ -z "$(ls -A $HOME/.cache/huggingface 2>/dev/null)" ]; then
    echo "警告: HuggingFace缓存目录为空，模型可能未正确下载"
    echo "如果首次运行失败，请检查模型下载过程"
fi

# 设置环境变量（通过环境变量控制是否开启轻量模式，默认关闭以获得更高精度）
export LIGHTWEIGHT_MODE=${LIGHTWEIGHT_MODE:-false}
export SESSION_DB_PATH=${SESSION_DB_PATH:-/app/data/session_store.db}

# 设置 pyrender 使用 EGL（无头渲染，适合 Docker 容器）
export PYOPENGL_PLATFORM=egl

echo "环境变量:"
echo "  LIGHTWEIGHT_MODE: $LIGHTWEIGHT_MODE"
echo "  SESSION_DB_PATH: $SESSION_DB_PATH"
echo ""

# 执行传入的命令
exec "$@"




