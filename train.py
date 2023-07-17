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


def save_traced_model(model: nn.Module, save_filename: Path, *, dataset: Dataset) -> Path:
    test_dataloader = DataLoader(dataset, batch_size=512)
    x, _ = next(iter(test_dataloader))
    traced = torch.jit.trace(model.net, x)
    save_filename.parent.mkdir(exist_ok=True)
    traced.save(save_filename)
    return save_filename


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
    trainer = pl.Trainer(enable_checkpointing=False, accelerator=device, devices=1, max_epochs=configs.num_epochs)
    trainer.fit(model=digit_recognizer, train_dataloaders=train_dataloader)
    print("Saving model...")
    model_path = MODEL_DIR_PATH / "model.pt"
    save_traced_model(digit_recognizer, model_path, dataset=train_ds)

    # Evaluate model
    print("Evaluating model...")
    trainer.test(model=digit_recognizer, dataloaders=train_dataloader)
    print("Done!")
