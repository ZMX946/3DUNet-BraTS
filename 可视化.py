import torch
import numpy as np
import matplotlib.pyplot as plt

from models.unet import UNet
from models.blocks import PlainBlock, ResidualBlock
from dataset.dataset_utils import RobustZScoreNormalization
import monai.transforms as transforms
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from configs import *


# ====================== 模型加载 ======================
block_dict = {
    'plain': PlainBlock,
    'res': ResidualBlock
}

args = parse_seg_args()

kwargs = {
    "input_channels": args.input_channels,
    "output_classes": args.num_classes,   # = 3
    "channels_list": args.channels_list,
    "deep_supervision": args.deep_supervision,
    "ds_layer": args.ds_layer,
    "kernel_size": args.kernel_size,
    "dropout_prob": args.dropout_prob,
    "norm_key": args.norm,
    "block": block_dict[args.block],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = UNet(**kwargs)

ckpt_path = "./exps_BraTS2021_Training_Data_unet_adamw_none_pos1.0_neg1.0_1212_093518/best_model_epoch_79.pth"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model.load_state_dict(ckpt["model"])
model = model.to(device).eval()

print("Model loaded")


# ====================== 数据加载 ======================
infer_transforms = transforms.Compose([
    transforms.LoadImaged(keys=['flair', 't1', 't1ce', 't2', 'label']),
    transforms.EnsureChannelFirstd(keys=['flair', 't1', 't1ce', 't2', 'label']),
    transforms.Orientationd(keys=['flair', 't1', 't1ce', 't2', 'label'], axcodes="RAS"),
    RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),
    transforms.ConcatItemsd(keys=['flair', 't1', 't1ce', 't2'], name='image', dim=0),
    transforms.DeleteItemsd(keys=['flair', 't1', 't1ce', 't2']),
    transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
    transforms.DivisiblePadd(keys=["image", "label"], k=16),
    transforms.EnsureTyped(keys=["image", "label"], device=device),
])

case = "BraTS2021_00000"
case_dir = rf"E:\study_code\智能体项目\BraTS2021数据\BraTS2021_Training_Data\{case}"

data = {
    "flair": f"{case_dir}/{case}_flair.nii.gz",
    "t1":    f"{case_dir}/{case}_t1.nii.gz",
    "t1ce":  f"{case_dir}/{case}_t1ce.nii.gz",
    "t2":    f"{case_dir}/{case}_t2.nii.gz",
    "label": f"{case_dir}/{case}_seg.nii.gz",
}

batch = infer_transforms(data)

image = batch["image"]        # (4, D, H, W)
label_mc = batch["label"]     # (3, D, H, W)

print("Data loaded:", image.shape)


# ====================== 推理（✔ 正确方式） ======================
with torch.no_grad():
    logits = model(image.unsqueeze(0))     # (1, 3, D, H, W)
    probs = torch.sigmoid(logits)[0]        # (3, D, H, W)

# 阈值（二值化）
pred_mc = (probs > 0.5).cpu().numpy()       # (3, D, H, W)


# ====================== 构造单通道 label（仅用于可视化） ======================
pred = np.zeros(pred_mc.shape[1:], dtype=np.uint8)

pred[pred_mc[0] == 1] = 2    # WT
pred[pred_mc[1] == 1] = 1    # TC 覆盖 WT
pred[pred_mc[2] == 1] = 3    # ET 覆盖 TC

label = np.zeros_like(pred)
label[label_mc[0].cpu().numpy() == 1] = 2
label[label_mc[1].cpu().numpy() == 1] = 1
label[label_mc[2].cpu().numpy() == 1] = 3

print("Inference done")


# ====================== Dice（✔ 与训练一致） ======================
def dice_score(pred, gt, eps=1e-5):
    inter = np.sum(pred & gt)
    return (2 * inter + eps) / (pred.sum() + gt.sum() + eps)

dice = {}
dice["WT"] = dice_score(pred_mc[0], label_mc[0].cpu().numpy())
dice["TC"] = dice_score(pred_mc[1], label_mc[1].cpu().numpy())
dice["ET"] = dice_score(pred_mc[2], label_mc[2].cpu().numpy())

print("Dice:", dice)


# ====================== 2D 可视化 ======================
img_np = image.cpu().numpy()

def show_slice(slice_idx):
    flair = img_np[0, slice_idx]

    plt.figure(figsize=(16,4), dpi=600)

    plt.subplot(1,4,1)
    plt.imshow(flair, cmap="gray")
    plt.title("FLAIR"); plt.axis("off")

    plt.subplot(1,4,2)
    plt.imshow(label[slice_idx], cmap="tab10", vmin=0, vmax=3)
    plt.title("GT"); plt.axis("off")

    plt.subplot(1,4,3)
    plt.imshow(pred[slice_idx], cmap="tab10", vmin=0, vmax=3)
    plt.title("Pred"); plt.axis("off")

    plt.subplot(1,4,4)
    plt.imshow(flair, cmap="gray")
    plt.imshow(pred[slice_idx], cmap="jet", alpha=0.4)
    plt.title("Overlay"); plt.axis("off")

    plt.tight_layout()
    plt.savefig(f"{case}_{slice_idx:03d}.pdf")
    plt.close()

for s in range(50, 100, 5):
    show_slice(s)


# ====================== 轮廓 ======================
def show_contour(slice_idx):
    plt.figure(figsize=(6,6), dpi=600)
    plt.imshow(img_np[0, slice_idx], cmap="gray")

    for cls, color in [(1,'yellow'), (2,'cyan'), (3,'red')]:
        plt.contour(label[slice_idx] == cls, colors=color, linewidths=1)
        plt.contour(pred[slice_idx] == cls, colors='white', linewidths=1)

    plt.axis("off")
    plt.savefig(f"{case}_{slice_idx:03d}_contour.pdf")
    plt.close()

show_contour(80)


# ====================== 3D 可视化（ET） ======================
def show_3d_mask(mask, color, name):
    if mask.sum() == 0:
        print(f"[SKIP] {name} empty")
        return

    verts, faces, _, _ = measure.marching_cubes(
        mask.astype(np.uint8), level=0.5
    )

    mesh = Poly3DCollection(verts[faces], alpha=0.6)
    mesh.set_facecolor(color)
    mesh.set_edgecolor('none')

    fig = plt.figure(figsize=(8,8), dpi=600)
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(mesh)

    # 🔥 核心：保持真实 3D 比例
    ax.set_box_aspect(mask.shape[::-1])

    ax.set_xlim(0, mask.shape[2])
    ax.set_ylim(0, mask.shape[1])
    ax.set_zlim(0, mask.shape[0])

    ax.view_init(elev=20, azim=45)
    ax.set_axis_off()

    plt.title(name)
    plt.tight_layout()
    plt.savefig(f"{case}_3d_{name}.pdf")
    plt.close()


show_3d_mask(pred_mc[2], 'red', 'ET')


# ====================== 失败分析 ======================
unique, counts = np.unique(pred, return_counts=True)
print("Pred distribution:", dict(zip(unique, counts)))

if dice["ET"] < 0.3:
    print("WARNING: ET Dice very low")

if label_mc[2].sum() > 0 and pred_mc[2].sum() == 0:
    print("ERROR: ET completely missed")

if dice["WT"] > 0.7 and dice["TC"] < 0.4:
    print("WARNING: Model biased to edema")
