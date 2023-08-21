from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Config(BaseModel):
    img_size: int = 224
    optimizer: str = 'adam'
    num_workers: int = 1
    learning_rate: float = 1e-5
    learning_rate_decay_step: int = 2
    learning_rate_decay_rate: float = 0.9
    momentum: float = 0.9
    weight_decay: float = 1e-5
    batch_size: int = 32
    epochs: int = 80
    age_stddev: float = 1.0
    data_dir: Path = Path('/mnt/data/age-estimation')
    resume_path: Path = Path('/mnt/model/epoch044_0.02343_3.9984.pth')
    checkpoint: Path = Path('/mnt/model')
