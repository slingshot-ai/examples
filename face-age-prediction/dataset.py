# Reference: Code modified from "https://github.com/yu4u/age-estimation-pytorch/tree/master"
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class FaceDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        dataset_split_type: str,  # to select the correct csv file for either train, validation, or test
        img_size: int = 224,  # size of the image to be fed to the model
        augment: bool = False,  # only augment the training dataset
        age_stddev: float = 1.0,  # standard deviation of the gaussian noise added to the age. This value is present in the dataset per image.
    ):
        assert dataset_split_type in {"train", "valid", "test"}
        csv_path = Path(data_dir).joinpath(f"gt_avg_{dataset_split_type}.csv")
        img_dir = Path(data_dir).joinpath(dataset_split_type)
        self.img_size = img_size
        self.augment = augment
        self.age_stddev = age_stddev

        transforms_aug_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        transforms_aug_val = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        if augment:
            self.transform = transforms_aug_train
        else:
            self.transform = transforms_aug_val

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))

        for _, row in df.iterrows():
            img_name = row["file_name"]

            img_path = img_dir.joinpath(img_name + "_face.jpg")
            assert img_path.is_file()
            self.x.append(str(img_path))
            self.y.append(row["apparent_age_avg"])
            self.std.append(row["apparent_age_std"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx: int):
        img_path = self.x[idx]
        age = self.y[idx]

        if self.augment:
            age += np.random.randn() * self.std[idx] * self.age_stddev

        img = cv2.imread(str(img_path), 1)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = self.transform(img)

        return img, np.clip(round(age), 0, 100)