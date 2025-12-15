import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

def robust_sigmoid(x):
    return torch.clamp(torch.sigmoid(x), min=0.0, max=1.0)


def sum_tensor(inp, axes, keepdim=False):
    axes = np.unique(axes).astype(int)
    if keepdim:
        for ax in axes:
            inp = inp.sum(int(ax), keepdims=True)
    else:
        for ax in sorted(axes, reverse=True):
            inp = inp.sum(int(ax))
    return inp


def get_tp_fp_fn_tn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):    # gt (b, x, y(, z))
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))  # gt (b, 1, ...)

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x, device=net_output.device)
            y_onehot.scatter_(1, gt, 1)  # (b, 1, ...) -> (b, c, ...)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot
    tn = (1 - net_output) * (1 - y_onehot)

    if mask is not None:
        tp = torch.stack([x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)], dim=1)
        fp = torch.stack([x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)], dim=1)
        fn = torch.stack([x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)], dim=1)
        tn = torch.stack([x_i * mask[:, 0] for x_i in torch.unbind(tn, dim=1)], dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2
        tn = tn ** 2

    if len(axes) > 0:
        tp = sum_tensor(tp, axes, keepdim=False)
        fp = sum_tensor(fp, axes, keepdim=False)
        fn = sum_tensor(fn, axes, keepdim=False)
        tn = sum_tensor(tn, axes, keepdim=False)

    return tp, fp, fn, tn


# =================== Weighted Dice + BCE for BraTS ===================
class BraTSWeightedDiceBCE(nn.Module):
    """
    给 BraTS 三个类别加权的 Dice + BCE Loss
    dice_weights: [WT, TC, ET]
    bce_weights:  [WT, TC, ET]
    """
    def __init__(self, dice_weights=(1.0, 2.0, 2.0), bce_weights=(0.5, 1.0, 1.5), smooth=1.0):
        super().__init__()
        self.dice_weights = torch.tensor(dice_weights).float()
        self.bce_weights = torch.tensor(bce_weights).float()
        self.smooth = smooth
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, net_output: Tensor, target: Tensor):
        # net_output: [B, C, ...]
        # target: [B, C, ...]

        probs = torch.sigmoid(net_output)
        axes = tuple(range(2, probs.ndim))

        # ---------- Dice ----------
        tp, fp, fn, _ = get_tp_fp_fn_tn(probs, target, axes)
        dice_per_class = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)

        w_dice = self.dice_weights.to(dice_per_class.device)
        dice_loss = 1.0 - (dice_per_class * w_dice).sum() / w_dice.sum()

        # ---------- BCE ----------
        bce_raw = self.bce(net_output, target)  # [B, C, ...]
        bce_per_class = bce_raw.mean(dim=tuple(range(2, bce_raw.ndim)))  # [B, C]

        w_bce = self.bce_weights.to(bce_per_class.device)
        bce_loss = (bce_per_class * w_bce).sum() / w_bce.sum()

        return bce_loss, dice_loss
