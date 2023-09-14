import os
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
import torch
from dataset import FaceDataset
from lightning.pytorch.loggers import WandbLogger
from model import FaceAgeTrainer
from pydantic import BaseModel
from slingshot.sdk.utils import get_config
from torch.utils.data import DataLoader


class TrainConfig(BaseModel):
    img_size: int = 224
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 20
    dataset_path: Path = Path("/mnt/dataset")
    output_path: Path = Path("/mnt/model")


# These are long lines
def load_dataset(
    data_path: Path,
    split: Literal["train", "valid", "test"],
    img_size: int,
    batch_size: int,
    num_workers: int = 0,
    augment: bool = False,
) -> DataLoader:
    # Setting up the training dataset and dataloader
    dataset = FaceDataset(data_path, split, img_size=img_size, augment=augment)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"), num_workers=num_workers  # Only shuffle for training
    )
    return dataloader


if __name__ == "__main__":
    config_train = get_config(TrainConfig)
    print(f"Config: {config_train.model_dump_json(indent=2)}")

    # Create model
    torch.random.manual_seed(0)
    device = "gpu" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    train_dataloader = load_dataset(
        config_train.dataset_path,
        split="train",
        img_size=config_train.img_size,
        batch_size=config_train.batch_size,
        augment=True,
    )
    val_dataloader = load_dataset(
        config_train.dataset_path,
        split="valid",
        img_size=config_train.img_size,
        batch_size=config_train.batch_size,
        augment=False,
    )
    print(f"Train dataset size: {len(train_dataloader.dataset)}")
    print(f"Training with batch size: {config_train.batch_size}")
    print(f"Validation dataset size: {len(val_dataloader.dataset)}")

    # Create model
    model = FaceAgeTrainer(learning_rate=config_train.learning_rate)

    # Train model
    trainer = pl.Trainer(
        default_root_dir=config_train.output_path,
        accelerator=device,
        devices=1,
        max_epochs=config_train.epochs,
        logger=WandbLogger() if os.environ.get("WANDB_API_KEY") else None,
    )
    trainer.fit(model=model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
    # Save the model
    print("Saving model...")
    trainer.save_checkpoint(config_train.output_path / "model.ckpt")

    print("Done!")
