# Docker 部署指南

本指南将帮助您使用Docker快速部署SAM-3D-Body应用，所有模型文件已预打包在镜像中，开箱即用。

## 前置要求

1. **Docker** (版本 20.10+)
   - Windows/Mac: 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop)
   - Linux: 安装 Docker Engine

2. **NVIDIA Docker支持** (推荐，用于GPU加速)
   - 安装 [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
   - 需要NVIDIA驱动和CUDA支持

3. **系统要求**
   - 至少 8GB 可用内存
   - 如果使用GPU: 至少 8GB VRAM
   - 至少 10GB 可用磁盘空间（用于镜像和模型）

## 快速开始

### 方法一: 使用构建脚本（推荐）

#### Linux/Mac:
```bash
chmod +x build-docker.sh
./build-docker.sh
```

#### Windows:
```cmd
build-docker.bat
```

### 方法二: 手动构建

```bash
docker build -t sam-3d-body:latest .
```

**注意**: 首次构建会下载所有模型文件（约2GB），可能需要10-30分钟，取决于网络速度。

## 运行容器

### 使用 Docker Compose（推荐）

```bash
docker-compose up -d
```

访问应用: http://localhost:5000

### 使用 Docker 命令

#### GPU模式（推荐）:
```bash
docker run -d \
  --name sam-3d-body \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  sam-3d-body:latest
```

#### CPU模式（较慢）:
```bash
docker run -d \
  --name sam-3d-body \
  -p 5000:5000 \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/outputs:/app/outputs \
  -v $(pwd)/data:/app/data \
  sam-3d-body:latest
```

## 环境变量

可以通过环境变量配置应用行为:

```bash
# 轻量级模式（减少VRAM使用，禁用FOV估计器）
LIGHTWEIGHT_MODE=true

# 自定义会话数据库路径
SESSION_DB_PATH=/app/data/session_store.db
```

### 在docker-compose.yml中设置:

```yaml
environment:
  - LIGHTWEIGHT_MODE=false
  - SESSION_DB_PATH=/app/data/session_store.db
```

### 在docker run命令中设置:

```bash
docker run -d \
  -e LIGHTWEIGHT_MODE=true \
  -p 5000:5000 \
  sam-3d-body:latest
```

## 数据持久化

容器使用以下卷来持久化数据:

- `./uploads` - 上传的图片文件
- `./outputs` - 处理结果和rig数据
- `./data` - 会话数据库

这些目录会在容器启动时自动创建。

## 查看日志

```bash
# 使用docker-compose
docker-compose logs -f

# 使用docker命令
docker logs -f sam-3d-body
```

## 停止和删除容器

```bash
# 停止容器
docker-compose down
# 或
docker stop sam-3d-body

# 删除容器
docker-compose down -v
# 或
docker rm sam-3d-body
```

## 故障排除

### 1. GPU不可用

**问题**: 容器无法使用GPU

**解决方案**:
- 确保安装了NVIDIA Container Toolkit
- 检查NVIDIA驱动: `nvidia-smi`
- 测试GPU访问: `docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi`

### 2. 模型加载失败

**问题**: 应用启动时模型加载失败

**解决方案**:
- 检查容器日志: `docker logs sam-3d-body`
- 确保构建时模型已正确下载
- 如果模型未下载，可以重新构建镜像

### 3. 内存不足

**问题**: 容器因内存不足而崩溃

**解决方案**:
- 使用轻量级模式: `LIGHTWEIGHT_MODE=true`
- 增加Docker内存限制
- 确保系统有足够的可用内存

### 4. 端口被占用

**问题**: 端口5000已被占用

**解决方案**:
- 修改docker-compose.yml中的端口映射: `"8080:5000"`
- 或使用docker run时指定: `-p 8080:5000`

### 5. 构建失败

**问题**: Docker构建过程中失败

**解决方案**:
- 检查网络连接（需要下载模型）
- 确保有足够的磁盘空间
- 查看详细构建日志: `docker build --progress=plain .`

## 性能优化

### GPU模式
- 确保使用 `--gpus all` 参数
- 检查GPU使用情况: `nvidia-smi`

### 轻量级模式
如果GPU内存有限（8GB或更少），启用轻量级模式:
```bash
LIGHTWEIGHT_MODE=true docker-compose up -d
```

### 模型缓存
如果需要持久化模型缓存（避免重新下载），取消注释docker-compose.yml中的卷映射:
```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
```

## 镜像大小

构建后的镜像大小约为:
- 基础镜像: ~2GB
- 模型文件: ~2-3GB
- 依赖和代码: ~1GB
- **总计**: ~5-6GB

## 更新镜像

```bash
# 重新构建
docker build -t sam-3d-body:latest .

# 停止旧容器
docker-compose down

# 启动新容器
docker-compose up -d
```

## 生产环境部署

### 使用反向代理（Nginx）

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

### 使用HTTPS

建议使用Let's Encrypt或类似服务配置HTTPS。

## 常见问题

**Q: 镜像构建需要多长时间？**
A: 首次构建需要10-30分钟，取决于网络速度和系统性能。后续构建会更快（使用缓存）。

**Q: 可以在没有GPU的情况下运行吗？**
A: 可以，但处理速度会显著降低。建议至少使用CPU模式进行测试。

**Q: 如何更新模型？**
A: 重新构建Docker镜像，模型会在构建时自动更新。

**Q: 数据会丢失吗？**
A: 不会，只要正确配置了卷映射，数据会持久化在主机上。

## 支持

如果遇到问题，请:
1. 查看容器日志
2. 检查系统资源（内存、磁盘、GPU）
3. 查看GitHub Issues
4. 提交新的Issue（包含日志和系统信息）




