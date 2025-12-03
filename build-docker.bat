@echo off
REM Docker镜像构建脚本 (Windows版本)

echo ==========================================
echo SAM-3D-Body Docker 镜像构建脚本
echo ==========================================

REM 检查Docker是否安装
where docker >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未找到Docker，请先安装Docker Desktop
    exit /b 1
)

REM 设置镜像名称和标签
set IMAGE_NAME=sam-3d-body
set IMAGE_TAG=%1
if "%IMAGE_TAG%"=="" set IMAGE_TAG=latest

echo.
echo 构建参数:
echo   镜像名称: %IMAGE_NAME%:%IMAGE_TAG%
echo   构建上下文: %CD%
echo.

REM 询问用户是否继续
set /p REPLY="是否开始构建? (y/n) "
if /i not "%REPLY%"=="y" (
    echo 构建已取消
    exit /b 1
)

echo.
echo 开始构建Docker镜像...
echo 注意: 首次构建会下载所有模型文件，可能需要较长时间（~2GB）
echo.

REM 构建镜像
docker build --tag %IMAGE_NAME%:%IMAGE_TAG% --tag %IMAGE_NAME%:latest --progress=plain .

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ==========================================
    echo ✓ 镜像构建成功！
    echo ==========================================
    echo.
    echo 镜像信息:
    docker images %IMAGE_NAME%:%IMAGE_TAG%
    echo.
    echo 运行镜像:
    echo   docker run -d -p 5000:5000 --gpus all %IMAGE_NAME%:%IMAGE_TAG%
    echo.
    echo 或使用docker-compose:
    echo   docker-compose up -d
    echo.
) else (
    echo.
    echo ==========================================
    echo ✗ 镜像构建失败
    echo ==========================================
    exit /b 1
)




