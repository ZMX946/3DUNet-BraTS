import matplotlib
matplotlib.use('Agg')  # 无显示环境下保存图片
import matplotlib.pyplot as plt
import numpy as np
import os
import torch

def _to_numpy(x):
    """把 tensor 转到 cpu 并转为 numpy（保证 dtype=float32 或 uint8）"""
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return x

def _get_center_slices(img):
    """返回轴向(Axial), 冠状(Coronal), 矢状(Sagittal)三视图的中心切片索引及切片"""
    # img shape: (C, D, H, W) 或 (D, H, W) 或 (N, C, D, H, W)
    if img.ndim == 5:  # (N, C, D, H, W)
        img = img[0]
    if img.ndim == 4:  # (C, D, H, W) -> take first channel
        img = img[0]
    # now img: (D, H, W)
    D, H, W = img.shape
    return {
        'axial': img[D // 2, :, :],
        'coronal': img[:, H // 2, :],
        'sagittal': img[:, :, W // 2]
    }

def save_prediction_visuals(save_dir, case_name, image, prediction, label=None, prefix='pred', logger=None):
    """
    保存三视图的叠加图
    Args:
        save_dir: str, 保存目录
        case_name: str, 病例名（无扩展）
        image: tensor/np (C,D,H,W) 或 (D,H,W) ，强度图（会做简单归一化）
        prediction: tensor/np (D,H,W) 二值分割 map
        label: (optional) 真值 mask，用于对比
        prefix: 文件名前缀
    """
    os.makedirs(save_dir, exist_ok=True)
    img = _to_numpy(image)
    pred = _to_numpy(prediction)
    if img.ndim == 4:  # (C,D,H,W)
        img = img[0]
    # 归一化图像到 [0,1]
    img_min, img_max = img.min(), img.max()
    img_norm = (img - img_min) / (img_max - img_min + 1e-8)

    slices = _get_center_slices(img_norm)
    pred_slices = _get_center_slices(pred)
    label_slices = _get_center_slices(_to_numpy(label)) if label is not None else None

    for plane in ['axial', 'coronal', 'sagittal']:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=600)
        ax.imshow(slices[plane], cmap='gray', interpolation='nearest')
        # overlay prediction mask with alpha
        ax.imshow(pred_slices[plane], cmap='jet', alpha=0.35, interpolation='nearest')
        if label_slices is not None:
            # 用轮廓或半透明叠加真值（更明显）
            ax.contour(label_slices[plane], levels=[0.5], colors='lime', linewidths=1)
        ax.axis('off')
        out_path1 = os.path.join(save_dir, f"{case_name}_{prefix}_{plane}.png")
        out_path2 = os.path.join(save_dir, f"{case_name}_{prefix}_{plane}.pdf")
        logger.info(f"Saved {out_path1}")
        plt.tight_layout(pad=0)
        fig.savefig(out_path1, bbox_inches='tight', pad_inches=0)
        fig.savefig(out_path2, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
