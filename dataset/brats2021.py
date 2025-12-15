import os
from os.path import join

import torch
from torch.utils.data import Dataset, DataLoader
import monai.transforms as transforms

from dataset.dataset_utils import nib_load, RobustZScoreNormalization


####################################################################################################

# transforms

def get_brats2021_base_transform():
    base_transform = [
        transforms.LoadImaged(keys=['flair', 't1', 't1ce', 't2', 'label']),
        transforms.EnsureChannelFirstd(keys=['flair', 't1', 't1ce', 't2', 'label']),
        transforms.Orientationd(keys=['flair', 't1', 't1ce', 't2', 'label'], axcodes="RAS"),
        RobustZScoreNormalization(keys=['flair', 't1', 't1ce', 't2']),
        transforms.ConcatItemsd(keys=['flair', 't1', 't1ce', 't2'], name='image', dim=0),
        transforms.DeleteItemsd(keys=['flair', 't1', 't1ce', 't2']),
        transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys='label'),
    ]
    return base_transform


def get_brats2021_train_transform(args):
    base_transform = get_brats2021_base_transform()
    data_aug = [
        transforms.RandCropByPosNegLabeld(
            keys=["image", 'label'],
            label_key='label',
            spatial_size=[args.patch_size] * 3,
            pos=args.pos_ratio,
            neg=args.neg_ratio,
            num_samples=1
        ),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=0),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=1),
        transforms.RandFlipd(keys=["image", 'label'], prob=0.5, spatial_axis=2),
        transforms.RandGaussianNoised(keys='image', prob=0.15, mean=0.0, std=0.33),
        transforms.RandGaussianSmoothd(
            keys='image', prob=0.15,
            sigma_x=(0.5, 1.5), sigma_y=(0.5, 1.5), sigma_z=(0.5, 1.5)
        ),
        transforms.RandAdjustContrastd(keys='image', prob=0.15, gamma=(0.7, 1.3)),
        transforms.EnsureTyped(keys=["image", 'label']),
    ]
    seed = args.seed % (2 ** 32 - 1)  # 限制 seed 范围
    return transforms.Compose(base_transform + data_aug)


def get_brats2021_infer_transform(args):
    base_transform = get_brats2021_base_transform()
    infer_transform = [transforms.EnsureTyped(keys=["image", 'label'])]
    return transforms.Compose(base_transform + infer_transform)


####################################################################################################

# dataset

class BraTS2021Dataset(Dataset):
    def __init__(self, data_root: str, mode: str, case_names: list = [], transforms=None):  # 修正：添加双下划线
        super(BraTS2021Dataset, self).__init__()
        assert mode in ['train', 'infer'], 'Unknown mode'
        self.mode = mode
        self.data_root = data_root
        self.case_names = case_names
        self.transforms = transforms

    def __getitem__(self, index: int):
        name = self.case_names[index]
        base_dir = join(self.data_root, name, name)
        data_dict = {
            'flair': base_dir + '_flair.nii.gz',
            't1': base_dir + '_t1.nii.gz',
            't1ce': base_dir + '_t1ce.nii.gz',
            't2': base_dir + '_t2.nii.gz',
            'label': base_dir + '_seg.nii.gz',
        }
        item = self.transforms(data_dict)

        # 如果是训练模式，RandCropByPosNegLabeld 返回 list
        if self.mode == 'train' and isinstance(item, list):
            item = item[0]

        # 确保 image 和 label 都是 tensor
        image = item['image']
        label = item['label']

        # MONAI 有时返回 list，强制转换
        if isinstance(image, list):
            image = image[0]
        if isinstance(label, list):
            label = label[0]

        # 转成 torch.Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image)
        if not isinstance(label, torch.Tensor):
            label = torch.as_tensor(label)

        return image, label, index, name

    def __len__(self):
        return len(self.case_names)


####################################################################################################

# dataloaders

def get_train_loader(args, case_names: list):
    train_transforms = get_brats2021_train_transform(args)
    train_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='train',
        case_names=case_names,
        transforms=train_transforms
    )
    return DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=True
    )


def get_infer_loader(args, case_names: list):
    infer_transforms = get_brats2021_infer_transform(args)
    infer_dataset = BraTS2021Dataset(
        data_root=os.path.join(args.data_root, args.dataset),
        mode='infer',
        case_names=case_names,
        transforms=infer_transforms
    )
    return DataLoader(
        infer_dataset,
        batch_size=args.infer_batch_size,
        shuffle=False,
        drop_last=False, num_workers=args.num_workers,
        pin_memory=True
    )
