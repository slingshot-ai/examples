import os
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
import torchvision

from pydantic import BaseModel
from model import DigitRecognizer
from torch import nn
from torch.utils.data import DataLoader, Dataset

DATASET_PATH = Path("/mnt/dataset")
MODEL_DIR_PATH = Path("/mnt/output")


class TrainConfig(BaseModel):
    batch_size: int = 256
    num_epochs: int = 2
    learning_rate: float = 0.01
    loss_fn: Literal["cross_entropy", "mse"] = "cross_entropy"


if __name__ == "__main__":
    print("Reading config...")
    configs = TrainConfig.parse_raw(os.environ.get("CONFIG", "{}"))

    # Create model
    torch.random.manual_seed(0)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    digit_recognizer = DigitRecognizer(configs.loss_fn, configs.learning_rate)

    # Load data
    print("Loading data...")
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5,), (0.5,))
        ]
    )
    train_ds = torchvision.datasets.MNIST(DATASET_PATH, transform=transform)
    print(f"Loaded {len(train_ds)} examples")
    train_dataloader = DataLoader(train_ds, batch_size=configs.batch_size)

    # Train model
    print("Training model...")
    trainer = pl.Trainer(default_root_dir=MODEL_DIR_PATH, accelerator=device, devices=1, max_epochs=configs.num_epochs)
    trainer.fit(model=digit_recognizer, train_dataloaders=train_dataloader)
    # Save the model
    print("Saving model...")
    trainer.save_checkpoint(MODEL_DIR_PATH / "model.ckpt")

    # Evaluate model
    print("Evaluating model...")
    trainer.test(model=digit_recognizer, dataloaders=train_dataloader)
    print("Done!")
