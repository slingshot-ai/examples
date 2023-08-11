from __future__ import annotations

from pydantic import BaseModel, Field, validator


class Config(BaseModel):
    # TODO: refactor variable names to be lowercase (i.e. remove aliases)
    img_size: int = Field(224)
    optimizer: str = Field('adam')
    num_workers: int = Field(1)
    learning_rate: float = Field(1e-5)
    learning_rate_decay_step: int = Field(2)
    learning_rate_decay_rate: float = Field(0.9)
    momentum: float = Field(0.9)
    weight_decay: float = Field(1e-5)
    batch_size: int = Field(32)
    epochs: int = Field(80)
    age_stddev: float = Field(1.0)
    data_dir: str = Field('/mnt/data/age-estimation')
    resume_path: str = Field(
        '/mnt/model/epoch044_0.02343_3.9984.pth', alias='resume_path'
    )
    checkpoint: str = Field('/mnt/model', alias='checkpoint')
