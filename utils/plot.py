import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle

# =====================================================
# 1. UNet 结构图（Encoder + Decoder + Skip Connections）
# =====================================================
def plot_unet_structure(channels_list, save_path=None):
    """
    绘制 UNet 通道结构图（Encoder + Decoder + Skip Connections）
    channels_list: 例如 [32, 64, 128, 256, 320]
    save_path: 输出图片路径（可选）
    """

    num_levels = len(channels_list)

    plt.figure(figsize=(12, 6))
    ax = plt.gca()

    # 在图中绘制每一层的 encoder & decoder block
    for i, ch in enumerate(channels_list):
        # Encoder blocks
        rect = Rectangle((i, 3), 0.6, 0.5, linewidth=2, edgecolor='blue', facecolor='none')
        ax.add_patch(rect)
        ax.text(i + 0.3, 3.25, f"{ch}", ha='center', color='blue')

        # Decoder blocks (反向)
        rect = Rectangle((i, 1), 0.6, 0.5, linewidth=2, edgecolor='red', facecolor='none')
        ax.add_patch(rect)
        ax.text(i + 0.3, 1.25, f"{channels_list[::-1][i]}", ha='center', color='red')

    # Skip connections
    for i in range(num_levels):
        arrow = FancyArrowPatch(
            (i + 0.3, 3), (i + 0.3, 1.5),
            arrowstyle='->', mutation_scale=12, linewidth=1.5, color='gray'
        )
        ax.add_patch(arrow)

    # 外观设置
    ax.set_xlim(-0.5, num_levels + 0.5)
    ax.set_ylim(0, 4.2)
    ax.axis('off')
    plt.title("UNet Encoder–Decoder Channel Structure", fontsize=18)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches='tight')

    return plt


# =====================================================
# 2. 综合可视化接口（供 main 调用）
# =====================================================
def generate_comprehensive_visualization(save_dir, data_root, num_cases=3):
    """
    你主程序中调用的可视化接口。
    这里加入 UNet 结构图自动生成。
    """
    os.makedirs(os.path.join(save_dir, "comprehensive_visualization"), exist_ok=True)
    viz_dir = os.path.join(save_dir, "comprehensive_visualization")

    # 创建 UNet 结构图
    unet_fig_path = os.path.join(viz_dir, "unet_structure.png")
    plot_unet_structure([32, 64, 128, 256, 320], save_path=unet_fig_path)
    plt.close()

    # 你可以在此基础上继续扩展另外的可视化
    return {
        "unet_structure": unet_fig_path,
    }
