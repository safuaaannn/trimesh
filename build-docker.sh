#!/bin/bash
# Docker镜像构建脚本

set -e

echo "=========================================="
echo "SAM-3D-Body Docker 镜像构建脚本"
echo "=========================================="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: 未找到Docker，请先安装Docker"
    exit 1
fi

# 检查nvidia-docker是否可用（可选）
if command -v nvidia-docker &> /dev/null || docker info | grep -q nvidia; then
    echo "✓ 检测到NVIDIA Docker支持"
else
    echo "警告: 未检测到NVIDIA Docker支持，GPU可能无法使用"
    echo "建议安装nvidia-container-toolkit"
fi

# 设置镜像名称和标签
IMAGE_NAME="sam-3d-body"
IMAGE_TAG="${1:-latest}"

echo ""
echo "构建参数:"
echo "  镜像名称: ${IMAGE_NAME}:${IMAGE_TAG}"
echo "  构建上下文: $(pwd)"
echo ""

# 询问用户是否继续
read -p "是否开始构建? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "构建已取消"
    exit 1
fi

echo ""
echo "开始构建Docker镜像..."
echo "注意: 首次构建会下载所有模型文件，可能需要较长时间（~2GB）"
echo ""

# 构建镜像
docker build \
    --tag "${IMAGE_NAME}:${IMAGE_TAG}" \
    --tag "${IMAGE_NAME}:latest" \
    --progress=plain \
    .

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 镜像构建成功！"
    echo "=========================================="
    echo ""
    echo "镜像信息:"
    docker images "${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "运行镜像:"
    echo "  docker run -d -p 5000:5000 --gpus all ${IMAGE_NAME}:${IMAGE_TAG}"
    echo ""
    echo "或使用docker-compose:"
    echo "  docker-compose up -d"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "✗ 镜像构建失败"
    echo "=========================================="
    exit 1
fi




