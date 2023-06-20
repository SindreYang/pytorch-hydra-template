import os
import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms


class MyDataset(Dataset):
    def __init__(
            self,
            data_dir: str = "datasets/torch_datasets",
            train_val_test_split=(55_000, 5_000, 10_000),
            batch_size: int = 2,
            num_workers: int = 0,
            pin_memory: bool = False,
            seed: int = 1024,
            mode: str = "train",
            augmentation: bool = True,
            prefetch_factor: int =2,
            sample_size: int = 1024,
    ):
        super().__init__()
        path_list = []
        for root, _, files in os.walk(data_dir):
            for file_name in files:
                if "pts" in file_name:  # 获取pts结尾的点云作为输入
                    path_list.append(os.path.join(root, file_name))

        self.data_train, self.data_val, self.data_test = random_split(
            dataset=path_list,
            lengths=train_val_test_split,
            generator=torch.Generator().manual_seed(seed),
        )
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed
        self.data_augmentation = augmentation
        self.sample_size = sample_size
        self.prefetch_factor=prefetch_factor

        if mode == "train":
            self.datasets = self.data_train
        elif mode == "val":
            self.datasets = self.data_val
        else:
            self.datasets = self.data_test

    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, idx):
        sources_path = self.datasets[idx]
        # 获取对应的target路径
        target_path = sources_path.replace("sources", "targets").replace(".pts", ".seg")

        # 处理数据
        points_list = []
        with open(sources_path) as f:
            sources_data = f.readlines()
            for i in sources_data:
                points_list.append(list(map(float, i.strip().split())))
        point_set = np.vstack(points_list)

        with open(target_path) as f:
            target_data = f.readlines()
            labels = np.array(list(map(float, target_data)))

        # 数据统一化
        choice = np.random.choice(len(labels), self.sample_size, replace=True)
        point_set = point_set[choice, :]
        labels = labels[choice]

        # 归一化
        point_set = point_set - np.expand_dims(np.mean(point_set, axis=0), 0)  # 中心点
        dist = np.max(np.sqrt(np.sum(point_set ** 2, axis=1)), 0)
        point_set = point_set / dist  # 缩放

        # 数据增强
        if self.data_augmentation:
            theta = np.random.uniform(0, np.pi * 2)
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            point_set[:, [0, 2]] = point_set[:, [0, 2]].dot(rotation_matrix)  # 随机旋转
            point_set += np.random.normal(0, 0.02, size=point_set.shape)  # 随机抖动

        # 转换类型
        point_set = torch.from_numpy(point_set.astype(np.float32)).transpose(1, 0)
        labels = torch.from_numpy(np.array([labels]).astype(np.int64))-1  # 从0开始

        return point_set, labels

    def _init_fn(self, worker_id):
        # 固定随机数
        np.random.seed(self.seed + worker_id)

    def train_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            worker_init_fn=self._init_fn,
            prefetch_factor=self.prefetch_factor,
        )

