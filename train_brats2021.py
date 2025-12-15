import os
import time
import warnings
from copy import deepcopy
from os.path import join
from utils.huatu import _to_numpy, save_prediction_visuals
from utils.plot import generate_comprehensive_visualization
warnings.filterwarnings("ignore")

import numpy as np
if not hasattr(np, 'bool'):
    np.bool = bool
if not hasattr(np, 'int'):
    np.int = int
if not hasattr(np, 'float'):
    np.float = float
import torch
import torch.nn as nn
from monai.inferers import sliding_window_inference
from torch.cuda.amp import GradScaler, autocast

import utils.metrics as metrics
from configs import parse_seg_args
from dataset import brats2021
from models import get_unet
from utils.loss import BraTSWeightedDiceBCE
from utils.misc import (AverageMeter, CaseSegMetricsMeterBraTS, ProgressMeter, LeaderboardBraTS,
                        brats_post_processing, initialization, load_cases_split, save_brats_nifti)
from utils.optim import get_optimizer
from utils.scheduler import get_scheduler
from monai.utils import set_determinism

set_determinism(seed=0)

import random

SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


###########################
# 训练函数
###########################
def train(args, epoch, model, train_loader, loss_fn, optimizer, scheduler, scaler, writer, logger):
    model.train()
    data_time = AverageMeter('Data', ':6.3f')
    batch_time = AverageMeter('Time', ':6.3f')
    bce_meter = AverageMeter('BCE', ':.4f')
    dsc_meter = AverageMeter('Dice', ':.4f')
    loss_meter = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, bce_meter, dsc_meter, loss_meter],
        prefix=f"Train: [{epoch}]")

    end = time.time()
    for i, (image, label, _, _) in enumerate(train_loader):
        try:
            if isinstance(label, list):
                label = torch.stack(label)
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image, dtype=torch.float32)
            if not isinstance(label, torch.Tensor):
                label = torch.as_tensor(label, dtype=torch.float32)

            image, label = image.cuda(), label.float().cuda()
            bsz = image.size(0)
            data_time.update(time.time() - end)

            with autocast((args.amp) and (scaler is not None)):
                preds = model(image)

                # 深度监督处理
                if isinstance(preds, (list, tuple)):
                    if len(preds) == 1:
                        weights = [1.0]
                    elif len(preds) == 2:
                        weights = [0.6, 0.4]
                    elif len(preds) == 3:
                        weights = [0.5, 0.3, 0.2]
                    else:
                        main_weight = 0.5
                        rest_weight = (1.0 - main_weight) / (len(preds) - 1)
                        weights = [main_weight] + [rest_weight] * (len(preds) - 1)

                    total_bce_loss = 0
                    total_dsc_loss = 0
                    for j, pred_j in enumerate(preds):
                        bce, dsc = loss_fn(pred_j, label)
                        total_bce_loss += bce * weights[j]
                        total_dsc_loss += dsc * weights[j]
                    bce_loss = total_bce_loss
                    dsc_loss = total_dsc_loss
                else:
                    bce_loss, dsc_loss = loss_fn(preds, label)

                loss = bce_loss + dsc_loss

            optimizer.zero_grad()

            # 检查 NaN/Inf（在 GPU 上）
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                logger.error(f"NaN/Inf loss detected at epoch {epoch} batch {i}. Saving batch for debug.")
                debug_path = os.path.join(args.exp_dir, "debug_bad_batch.pth")
                torch.save({'image': image.detach().cpu(), 'label': label.detach().cpu()}, debug_path)
                raise ValueError("NaN/Inf loss")

            # backward 与 step（支持 AMP 的 unscale + clip）
            if args.amp and scaler is not None:
                scaler.scale(loss).backward()
                # 取消 scale 以便做梯度裁剪
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=12.0)
                optimizer.step()

            # 强制同步，便于及时暴露 cuda error 的真实位置
            torch.cuda.synchronize()

            # 把 MONAI MetaTensor 安全地转换为 float
            bce_val = float(bce_loss.detach().cpu())
            dsc_val = float(dsc_loss.detach().cpu())
            loss_val = float(loss.detach().cpu())

            bce_meter.update(bce_val, bsz)
            dsc_meter.update(dsc_val, bsz)
            loss_meter.update(loss_val, bsz)
            batch_time.update(time.time() - end)

            if (i == 0) or (i + 1) % args.print_freq == 0:
                progress.display(i + 1, logger)

            end = time.time()

        except RuntimeError as e:
            # 捕获 CUDA 相关运行时错误，记录更多 debug 信息并保存 batch
            logger.error(f"RuntimeError at epoch {epoch} batch {i}: {e}")
            # 保存当前 batch（cpu）便于离线复现
            bad_dir = os.path.join(args.exp_dir, "bad_batches")
            os.makedirs(bad_dir, exist_ok=True)
            torch.save({
                'image': image.detach().cpu(),
                'label': label.detach().cpu(),
                'preds': preds.detach().cpu() if isinstance(preds, torch.Tensor) else None,
                'exception': str(e)
            }, os.path.join(bad_dir, f"bad_batch_ep{epoch}_idx{i}.pth"))
            # 推荐打印当前显存使用情况
            try:
                logger.error(torch.cuda.memory_summary(device=None, abbreviated=True))
            except Exception:
                pass
            # 重新抛出以便外层能够看到 traceback（或注释下一行以继续训练）
            raise
        except Exception as e:
            logger.exception(f"Unexpected exception at epoch {epoch} batch {i}: {e}")
            raise

    if scheduler is not None:
        scheduler.step()


###########################
# 推理函数
###########################
def infer(args, epoch, model, infer_loader, writer, logger, mode='val', save_pred=True):
    model.eval()

    batch_time = AverageMeter('Time', ':6.3f')
    case_metrics_meter = CaseSegMetricsMeterBraTS()

    folder_dir = mode if epoch is None else f"{mode}_epoch_{epoch:02d}"
    save_path = join(args.exp_dir, folder_dir)
    os.makedirs(save_path, exist_ok=True)

    logger.info(f"========== Running {mode.upper()} Epoch {epoch} ==========")
    logger.info(f"Saving results to: {save_path}")
    logger.info(f"Total {len(infer_loader)} batches")

    with torch.no_grad():
        end = time.time()

        for i, batch_data in enumerate(infer_loader):
            if len(batch_data) == 2:
                image, label = batch_data
                brats_names = [f"{mode}_case_{i}_{j}" for j in range(image.shape[0])]
            else:
                image, label, _, brats_names = batch_data

            if isinstance(label, list):
                label = torch.stack(label)
            image = torch.as_tensor(image, dtype=torch.float32)
            label = torch.as_tensor(label, dtype=torch.float32)

            image, label = image.cuda(), label.bool().cuda()
            bsz = image.size(0)

            seg_map = sliding_window_inference(
                inputs=image,
                predictor=model,
                roi_size=args.patch_size,
                sw_batch_size=args.sw_batch_size,
                overlap=args.patch_overlap,
                mode=args.sliding_window_mode
            )

            if isinstance(seg_map, (list, tuple)):
                seg_map = seg_map[0]

            seg_map = (seg_map > 0.5)
            seg_map = brats_post_processing(seg_map)

            dice = metrics.dice(seg_map, label)
            hd95 = metrics.hd95(seg_map, label)

            case_metrics_meter.update(dice, hd95, brats_names, bsz)

            if save_pred:
                # 1) 保存 NIfTI（原有函数）
                save_brats_nifti(seg_map, brats_names, mode, args.data_root, save_path, logger)

                # 2) 同时保存 PNG 可视化（按病例）
                # seg_map: tensor (N, D, H, W) 或 (D,H,W) per case
                # image: 原始输入 tensor (N, C, D, H, W) 或类似
                # label: 原始真值 mask (N, D, H, W)
                # 注意：batch 中可能包含多个样本
                vis_dir = os.path.join(save_path, "visualizations")
                os.makedirs(vis_dir, exist_ok=True)

                # 确保为 numpy / cpu
                imgs_np = _to_numpy(image)  # shape (N,C,D,H,W) 或 (N,D,H,W)
                preds_np = _to_numpy(seg_map)  # (N,D,H,W)
                labels_np = _to_numpy(label) if label is not None else None

                for bi, name in enumerate(brats_names):
                    img_i = imgs_np[bi]
                    pred_i = preds_np[bi]
                    label_i = labels_np[bi] if labels_np is not None else None
                    save_prediction_visuals(vis_dir, name, img_i, pred_i, label=label_i, prefix='pred', logger= logger)

            torch.cuda.synchronize()
            batch_time.update(time.time() - end)
            end = time.time()

            # ⭐ 每 1 个 batch 打印一次
            if (i == 0) or (i + 1) % args.print_freq == 0:
                logger.info(
                    f"[{mode}] Batch {i+1}/{len(infer_loader)} | "
                    f"Dice: {dice.mean().item():.4f} | "
                    f"HD95: {hd95.mean().item():.4f} | "
                    f"Time: {batch_time.val:.3f}"
                )

    # ⭐ 输出单病例结果（TXT）
    case_metrics_meter.output(save_path)

    # ⭐ 输出平均指标
    infer_metrics = case_metrics_meter.mean()

    logger.info(f"========== {mode.upper()} Epoch {epoch} Finished ==========")
    for key, value in infer_metrics.items():
        logger.info(f"{mode}/{key}: {value:.4f}")

    if writer is not None:
        for key, value in infer_metrics.items():
            writer.add_scalar(f"{mode}/{key}", value, epoch)

    return infer_metrics



###########################
# 主程序 main()
###########################
def main():
    args = parse_seg_args()
    logger, writer = initialization(args)

    train_cases, val_cases, test_cases = load_cases_split(args.cases_split)
    train_loader = brats2021.get_train_loader(args, train_cases)
    val_loader = brats2021.get_infer_loader(args, val_cases)
    test_loader = brats2021.get_infer_loader(args, test_cases)

    model = get_unet(args).cuda()
    if args.data_parallel:
        model = nn.DataParallel(model)

    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)
    loss = BraTSWeightedDiceBCE().cuda()
    scaler = GradScaler() if args.amp else None

    if args.weight_path:
        logger.info("Loading checkpoint...")
        model_state = torch.load(args.weight_path)['model']
        model.load_state_dict(model_state)

    logger.info("Training starts...")
    best_model = {}
    val_leaderboard = LeaderboardBraTS()

    for epoch in range(args.epochs):
        train(args, epoch, model, train_loader, loss, optimizer, scheduler, scaler, writer, logger)

        if (epoch + 1) % args.eval_freq == 0:
            val_metrics = infer(args, epoch, model, val_loader, writer, logger, mode='val')
            val_leaderboard.update(epoch, val_metrics)
            best_model[epoch] = deepcopy(model.state_dict())

        torch.cuda.empty_cache()

    val_leaderboard.output(args.exp_dir)
    best_epoch = val_leaderboard.get_best_epoch()
    model.load_state_dict(best_model[best_epoch])
    # === 保存最终最佳模型（推荐） ===
    save_path = os.path.join(args.exp_dir, f"best_model_epoch_{best_epoch:02d}.pth")
    torch.save({"epoch": best_epoch, "model": model.state_dict()}, save_path)
    logger.info(f"Best model saved to: {save_path}")

    logger.info(f"Testing best epoch {best_epoch}...")
    infer(args, best_epoch, model, test_loader, writer, logger, mode='test', save_pred=True)

    ##############################
    #   ⭐ **自动生成可视化图像**
    ##############################
    logger.info("Generating visualizations...")

    test_results_dir = os.path.join(args.exp_dir, f"test_epoch_{best_epoch:02d}")

    generate_comprehensive_visualization(
        test_results_dir,
        os.path.join(args.data_root, args.dataset),
        num_cases=3
    )

    logger.info("Visualization saved to:")
    logger.info(os.path.join(test_results_dir, "comprehensive_visualization"))


if __name__ == '__main__':
    main()
