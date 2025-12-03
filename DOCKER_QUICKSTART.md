# Docker 快速开始指南

## 🚀 一键部署

### 1. 构建镜像

**Linux/Mac:**
```bash
chmod +x build-docker.sh
./build-docker.sh
```

**Windows:**
```cmd
build-docker.bat
```

**手动构建:**
```bash
docker build -t sam-3d-body:latest .
```

> ⚠️ **注意**: 首次构建会下载约2GB的模型文件，需要10-30分钟

### 2. 启动容器

**使用 Docker Compose (推荐):**
```bash
docker-compose up -d
```

**使用 Docker 命令:**
```bash
# GPU模式
docker run -d --name sam-3d-body --gpus all -p 5000:5000 sam-3d-body:latest

# CPU模式
docker run -d --name sam-3d-body -p 5000:5000 sam-3d-body:latest
```

### 3. 访问应用

打开浏览器访问: **http://localhost:5000**

## 📋 常用命令

```bash
# 查看日志
docker-compose logs -f

# 停止容器
docker-compose down

# 重启容器
docker-compose restart

# 查看容器状态
docker ps
```

## ⚙️ 配置选项

编辑 `docker-compose.yml` 中的环境变量:

```yaml
environment:
  - LIGHTWEIGHT_MODE=false  # 设为true可减少VRAM使用
```

## 🔧 故障排除

**GPU不可用?**
- 确保安装了 NVIDIA Container Toolkit
- 检查: `nvidia-smi`

**端口被占用?**
- 修改 `docker-compose.yml` 中的端口映射

**内存不足?**
- 设置 `LIGHTWEIGHT_MODE=true`

更多详细信息请查看 [DOCKER_README.md](DOCKER_README.md)




