import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
from bidirectional_mamba import FusionModel
from train_and_test import train, test, val
from dataset import (
    NTU_HAR_Dataset,
    UT_HAR_dataset,
    XRF55Dataset,
    SSHARDataset,
    compute_sshar_mean_std,
    compute_xrf55_statistics
)
import gc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main(args):
    # 数据集路径和参数设置
    if args.dataset == 'NTU-Fi_HAR':
        # NTU-Fi_HAR 数据集的路径
        train_root_dir = '/kaggle/input/datasets/phanvanhai/ntu-har/NTU-Fi_HAR/train_amp'
        test_root_dir = '/kaggle/input/datasets/phanvanhai/ntu-har/NTU-Fi_HAR/test_amp'

        # 实例化 NTU-Fi_HAR 数据集
        train_dataset = NTU_HAR_Dataset(root_dir=train_root_dir, modal='CSIamp')
        test_dataset = NTU_HAR_Dataset(root_dir=test_root_dir, modal='CSIamp')

        # 设置 FusionModel 参数
        depth = 2
        embed_dim = 342
        channels = 1000
        num_classes = 6
        in_channels = 342
        out_channels = 342
        kernel_size = 5
        groups = 19
    elif args.dataset == 'UT':
        # UT 数据集的路径
        root = '/kaggle/input/datasets/phanvanhai/ut-har'
        data = UT_HAR_dataset(root)

        train_set = torch.utils.data.TensorDataset(data['X_train'], data['y_train'])
        test_set = torch.utils.data.TensorDataset(torch.cat((data['X_val'], data['X_test']), 0),
                                                  torch.cat((data['y_val'], data['y_test']), 0))

        # 设置 FusionModel 参数
        depth = 8
        embed_dim = 90
        channels = 250
        num_classes = 7
        in_channels = 90
        out_channels = 90
        kernel_size = 5
        groups = 3
    elif args.dataset == 'XRF55':
        train_root_dir = '/kaggle/input/datasets/phanvanhai/xrf55-s1-bd/XRF_dataset'
        test_root_dir = '/kaggle/input/datasets/phanvanhai/xrf55-s1-bd/XRF_dataset'

        mean, std = compute_xrf55_statistics(
            file_path=root_dir,
            scene="dml",
        )

        train_dataset = XRF55Dataset(
            root_dir=train_root_dir,
            split='train',
            mean=mean,
            std=std,
        )
        test_dataset = XRF55Dataset(
            root_dir=test_root_dir,
            split='test',
            mean=mean,
            std=std,
        )

        depth = 2
        embed_dim = 270
        channels = 1000
        num_classes = 11
        in_channels = 270
        out_channels = 270
        kernel_size = 5
        groups = 15
    elif args.dataset == 'SSHAR':
        root_dir = '/kaggle/input/datasets/phanvanhai/csi-room-02-0707'
        mean, std = compute_sshar_mean_std(
            root_dir=root_dir,
            device='esp',
            signal='amp'
        )
        train_dataset = SSHARDataset(
            root_dir=root_dir,
            device='esp',
            signal='amp',
            split='train',
            mean=mean,
            std=std,
        )

        test_dataset = SSHARDataset(
            root_dir=root_dir,
            device='esp',
            signal='amp',
            split='test',
            mean=mean,
            std=std,
        )

        depth = 2
        channels = 1000
        kernel_size = 5

        # esp
        num_classes = 8
        embed_dim = 168
        in_channels = 168
        out_channels = 168
        groups = 7

        # # asus
        # embed_dim = 672
        # in_channels = 672
        # out_channels = 672
        # groups = 28
    else:
        raise ValueError("Unsupported dataset. Choose either 'NTU' or 'UT'.")

    # 创建 DataLoader
    if args.dataset != 'UT':
        train_loader = DataLoader(
            train_dataset,
            batch_size=32,
            shuffle=True,
            num_workers=4,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=32,
            shuffle=False,
            num_workers=4,
        )
    else:
        train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size=128,
            shuffle=True,
            drop_last=True,
            num_workers=4,
        )

        test_loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=128,
            shuffle=False,
            num_workers=4,
        )
   
    # 实例化融合模型
    fusion_model = FusionModel(depth=depth, embed_dim=embed_dim, channels=channels,
                               num_classes=num_classes, in_channels=in_channels,
                               out_channels=out_channels, kernel_size=kernel_size, groups = groups).to(device)
    criterion = nn.CrossEntropyLoss()

    # 训练和测试
    train_epoch = 100
    train(
        model=fusion_model,
        tensor_loader=train_loader,
        val_loader=test_loader,
        num_epochs=train_epoch,
        learning_rate=1e-4 if args.dataset == 'NTU-Fi_HAR' else 1e-3,
        criterion=criterion,
        device=device
    )

    gc.collect()
    torch.cuda.empty_cache()

    fusion_model.load_state_dict(
        torch.load("best_model.pth")
    )
    test(
        model=fusion_model,
        tensor_loader=test_loader,
        criterion=criterion,
        device=device
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model with different datasets.")
    parser.add_argument('--dataset', choices=['NTU-Fi_HAR', 'UT', 'XRF55', 'SSHAR'], required=True, help="Choose dataset: 'NTU' or 'UT'.")
    args = parser.parse_args()

    main(args)
