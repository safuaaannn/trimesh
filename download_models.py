#!/usr/bin/env python3
"""
模型下载脚本
在Docker构建时预下载所有必需的模型文件
"""
import os
import sys
from pathlib import Path

def download_sam3d_body_model():
    """下载SAM-3D-Body主模型"""
    print("=" * 60)
    print("正在下载 SAM-3D-Body 主模型...")
    print("=" * 60)
    
    try:
        from sam_3d_body.build_models import _hf_download
        
        # 下载模型到HuggingFace缓存目录
        repo_id = "facebook/sam-3d-body-dinov3"
        ckpt_path, mhr_path = _hf_download(repo_id)
        
        print(f"✓ SAM-3D-Body模型下载完成")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  MHR模型: {mhr_path}")
        return True
    except Exception as e:
        print(f"✗ SAM-3D-Body模型下载失败: {e}")
        return False


def download_detector_model():
    """下载Human Detector模型"""
    print("=" * 60)
    print("正在下载 Human Detector (VitDet) 模型...")
    print("=" * 60)
    
    try:
        # 这个模型会在首次使用时自动下载
        # 我们通过导入来触发下载
        from tools.build_detector import load_detectron2_vitdet
        
        # 尝试加载模型以触发下载
        print("正在初始化Detector...")
        detector = load_detectron2_vitdet()
        print("✓ Human Detector模型准备完成")
        return True
    except Exception as e:
        print(f"✗ Human Detector模型下载失败: {e}")
        print("  注意: 模型将在首次使用时自动下载")
        return False


def download_fov_estimator():
    """下载FOV Estimator模型"""
    print("=" * 60)
    print("正在下载 FOV Estimator (MoGe2) 模型...")
    print("=" * 60)
    
    try:
        from tools.build_fov_estimator import load_moge
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 下载模型
        moge_model = load_moge(device, path="Ruicheng/moge-2-vitl-normal")
        print("✓ FOV Estimator模型下载完成")
        return True
    except Exception as e:
        print(f"✗ FOV Estimator模型下载失败: {e}")
        print("  注意: 模型将在首次使用时自动下载")
        return False


def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("开始下载所有必需的模型文件...")
    print("=" * 60 + "\n")
    
    results = []
    
    # 下载SAM-3D-Body主模型
    results.append(("SAM-3D-Body", download_sam3d_body_model()))
    
    # 下载Human Detector
    results.append(("Human Detector", download_detector_model()))
    
    # 下载FOV Estimator
    results.append(("FOV Estimator", download_fov_estimator()))
    
    # 打印总结
    print("\n" + "=" * 60)
    print("模型下载总结:")
    print("=" * 60)
    
    for name, success in results:
        status = "✓ 成功" if success else "✗ 失败（将在首次使用时下载）"
        print(f"{name}: {status}")
    
    print("\n" + "=" * 60)
    print("模型下载完成！")
    print("=" * 60 + "\n")
    
    # 如果主模型下载失败，退出并报错
    if not results[0][1]:
        print("错误: SAM-3D-Body主模型下载失败，这是必需的模型！")
        sys.exit(1)


if __name__ == "__main__":
    main()




