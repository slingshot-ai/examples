from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from pydantic import BaseModel


class TrainConfig(BaseModel):
    # --- Data ---
    target_entity_image_dir: Path
    generic_entity_image_dir: Path | None = None  # If None, the prior preservation loss is not used.
    target_prompt: str = "A photo of sks person"  # Prompt for the target entity
    generic_prompt: str = "A photo of a person"  # Prompt for the generic entities
    image_extensions: str = ".jpg,.jpeg,.png,.heic"  # Comma-separated list of image extensions to look for. TODO: use list once slingshot supports it
    # --- Image preprocessing ---
    resolution: int = 512
    center_crop: bool = False
    # --- Training ---
    base_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    train_text_encoder: bool = True
    train_unet: bool = True
    learning_rate: float = 2e-6
    lr_scheduler: str = "constant_with_warmup"
    lr_warmup_steps: int = 0
    max_grad_norm: float = 1.0
    max_train_steps: int = 1000
    prior_preservation_weight: float = 1.0
    gradient_accumulation_steps: int = 1
    # --- Memory --- (The following config barely fits on a T4 (16GB))
    train_batch_size: int = 1
    gradient_checkpointing: bool = True
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"
    use_8bit_adam: bool = True
    # --- Evaluation & saving ---
    checkpoint_dir: Path
    save_all_checkpoints: bool = False
    save_n_steps: int = 200
    eval_n_steps: int = 100
    eval_n_generate_samples: int = 9
    eval_num_inference_steps: int = 25
    eval_use_dreambooth_prompts: bool = True
    # --- Others ---
    seed: int = 0
    wandb_project: str = "dreambooth"
    dry_run: bool = False
    logging_dir: Path = Path("/mnt/logs")

    @property
    def torch_dtype(self) -> torch.dtype:
        """The data type of the weights of the models that are *not* trained.

        This includes the VAE, the text encoder if train_text_encoder=False, and the U-Net if train_unet=False.
        Models that are trained uses float32 regardless of this setting.
        """
        return _mixed_precision_tag_to_torch_dtype(self.mixed_precision)

    @property
    def use_prior_preservation_loss(self) -> bool:
        """Whether to use the prior preservation loss."""
        if self.generic_entity_image_dir is not None:
            assert self.prior_preservation_weight > 0
        return self.generic_entity_image_dir is not None


class InferenceConfig(BaseModel):
    """Configuration for inference.

    Components of the text encoder and the U-Net are loaded from checkpoints, if they exist.
    Otherwise, they are loaded from the base model. Thus, you can run vanilla Stable Diffusion by setting
    `checkpoints_path` to an empty directory.
    """

    checkpoints_path: Path
    base_model_name_or_path: str = "runwayml/stable-diffusion-v1-5"
    mixed_precision: Literal["no", "fp16", "bf16"] = "fp16"

    @property
    def torch_dtype(self) -> torch.dtype:
        """The data type of the weights of all models."""
        return _mixed_precision_tag_to_torch_dtype(self.mixed_precision)


def _mixed_precision_tag_to_torch_dtype(tag: Literal["no", "fp16", "bf16"]) -> torch.dtype:
    match tag:
        case "no":
            return torch.float32
        case "fp16":
            return torch.float16
        case "bf16":
            return torch.bfloat16
        case _:
            raise ValueError(f"Invalid mixed_precision: {tag}")
