# Reference: Code modified from "https://github.com/yu4u/age-estimation-pytorch/tree/master"
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        dataset_split: Literal["train", "valid", "test"],
        img_size: int = 224,
        augment: bool = False,
    ):
        """
        Args:
            data_dir: Path to the directory containing the dataset.
            dataset_split: Which dataset split to use. One of "train", "val", or "test".
            img_size: Size of the image to be fed to the model. Images must be square.
            augment: Whether to artificially augment the dataset. Should only be used for training.
        """
        csv_path = data_dir / f"gt_avg_{dataset_split}.csv"
        img_dir = data_dir / dataset_split
        self.img_size = img_size
        self.augment = augment

        transform_list = [transforms.Resize([self.img_size, self.img_size])]
        if augment:
            transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
        self.transform = transforms.Compose(transform_list)

        self.samples = pd.read_csv(str(csv_path))
        self.samples["file_path"] = self.samples["file_name"].apply(lambda filename: img_dir / f"{filename}_face.jpg")
        # Verify all files exist
        assert all(self.samples["file_path"].apply(lambda s: s.is_file()))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, np.float32]:
        sample = self.samples.iloc[idx]
        age = sample["apparent_age_avg"]

        img = read_image(str(sample["file_path"])).float() / 255.0  # We need to normalize manually due to read_image
        img = self.transform(img)

        # Making type explicit
        clipped_age: np.float32 = np.clip(age, 0, 100).astype(np.float32)
        return img, clipped_age
