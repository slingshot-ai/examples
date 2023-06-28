import json
import os
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from model import DigitRecognizer
from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils import MNISTExample, base64_to_tensor

EXAMPLE_DATA_PATH = Path("/mnt/dataset") / "dataset.jsonl"
MODEL_DIR_PATH = Path("/mnt") / "output"

torch.random.manual_seed(0)
device = "gpu" if torch.cuda.is_available() else "cpu"


class CustomDataset(Dataset):
    def __init__(self, obj: List[MNISTExample]):
        self.obj = obj

    def __getitem__(self, idx):
        example = self.obj[idx]
        return base64_to_tensor(example.example), int(example.label)

    def __len__(self):
        return len(self.obj)


def save_traced_model(model: nn.Module, save_filename: Path, *, dataset: CustomDataset) -> Path:
    test_dataloader = DataLoader(dataset, batch_size=512)
    x, _ = next(iter(test_dataloader))
    traced = torch.jit.trace(model.net, x)
    save_filename.parent.mkdir(exist_ok=True)
    traced.save(save_filename)
    return save_filename


if __name__ == "__main__":
    print("Reading config...")
    configs = json.loads(os.environ["CONFIG"])
    BATCH_SIZE = configs.get("batch_size", 256)
    NUM_EPOCHS = configs.get("num_epochs", 2)
    LEARNING_RATE = configs.get("learning_rate", 0.01)
    LOSS_FN = configs.get("loss_fn", "cross_entropy")

    # Create model
    print(f"Using device: {device}")
    digit_recognizer = DigitRecognizer(LOSS_FN, LEARNING_RATE)

    # Load data
    print("Loading data...")
    with open(EXAMPLE_DATA_PATH) as f:
        data = [MNISTExample.parse_raw(line) for line in f]
    print(f"Loaded {len(data)} examples")
    train_ds = CustomDataset(data)
    train_dataloader = DataLoader(train_ds, batch_size=BATCH_SIZE)

    # Train model
    print("Training model...")
    trainer = pl.Trainer(enable_checkpointing=False, accelerator=device, devices=1, max_epochs=NUM_EPOCHS)
    trainer.fit(model=digit_recognizer, train_dataloaders=train_dataloader)
    print("Saving model...")
    model_path = MODEL_DIR_PATH / "model.pt"
    save_traced_model(digit_recognizer, model_path, dataset=train_ds)

    # Evaluate model
    print("Evaluating model...")
    trainer.test(model=digit_recognizer, dataloaders=train_dataloader)
    print("Done!")
