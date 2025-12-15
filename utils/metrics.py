import numpy as np
from medpy.metric import hd95 as hd95_medpy
from torch import Tensor


def dice(output:Tensor, target:Tensor, eps: float=1e-5) -> np.ndarray:
    """calculate multilabel batch dice"""
    target = target.float()
    num = 2 * (output * target).sum(dim=(2,3,4)) + eps
    den = output.sum(dim=(2,3,4)) + target.sum(dim=(2,3,4)) + eps
    dsc = num / den

    return dsc.cpu().numpy()


def hd95(output: Tensor, target: Tensor, spacing=None) -> np.ndarray:
    """
    output, target: [B, C, D, H, W] boolean tensors
    return: numpy array of shape (B, C)
    """
    # 转成 NumPy，并且用 uint8 替换 bool，避免 medpy 使用已废弃的 np.bool
    output = output.bool().cpu().numpy().astype(np.uint8)
    target = target.bool().cpu().numpy().astype(np.uint8)

    B, C = target.shape[:2]
    hd95_arr = np.zeros((B, C), dtype=np.float64)

    for b in range(B):
        for c in range(C):
            pred = output[b, c]
            gt = target[b, c]

            # 下面是三种特例处理
            if pred.sum() == 0 and gt.sum() == 0:
                hd95_arr[b, c] = 0.0
            elif pred.sum() > 0 and gt.sum() == 0:
                hd95_arr[b, c] = 373.1287
            elif pred.sum() == 0 and gt.sum() > 0:
                hd95_arr[b, c] = 373.1287
            else:
                hd95_arr[b, c] = hd95_medpy(pred, gt, voxelspacing=spacing)

    return hd95_arr

