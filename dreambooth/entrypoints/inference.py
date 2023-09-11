import base64
import json
import logging
from io import BytesIO

import torch
from diffusers import DDIMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from PIL import Image
from slingshot import InferenceModel
from slingshot.sdk.utils import get_config
from transformers import CLIPTextModel

from dreambooth.config import InferenceConfig

logger = logging.getLogger("dreambooth")


class Dreambooth(InferenceModel):
    config: InferenceConfig
    loaded_model: tuple[str, StableDiffusionPipeline] | None = None  # tuple of (checkpoint_id, pipeline)

    def generate_image(self, checkpoint: str, prompt: str, **kwargs) -> list[Image]:
        pipeline = self.load_checkpoint(checkpoint)  # dynamically load the specified checkpoint
        with torch.autocast("cuda", dtype=self.config.torch_dtype):
            return pipeline(prompt, **kwargs).images

    def load_checkpoint(self, checkpoint: str) -> StableDiffusionPipeline:
        if self.loaded_model is not None and self.loaded_model[0] == checkpoint:
            # No-op if we've already loaded this checkpoint
            return self.loaded_model[1]

        kwargs = {
            "scheduler": DDIMScheduler.from_pretrained(self.config.base_model_name_or_path, subfolder="scheduler"),
            "torch_dtype": self.config.torch_dtype,
            "device_map": "auto",
            "variant": "fp16" if self.config.mixed_precision == "fp16" else None,  # Saves bandwidth if using fp16
            "safety_checker": None,
            "requires_safety_checker": False,
        }
        checkpoint_path = self.config.checkpoints_path / checkpoint
        assert checkpoint_path.exists(), f"Checkpoint {checkpoint_path} does not exist"

        # Load the fine-tuned unet/text_encoder if it exists, otherwise the base version will be loaded
        if (unet_path := checkpoint_path / "unet").exists():
            kwargs["unet"] = UNet2DConditionModel.from_pretrained(
                unet_path, torch_dtype=self.config.torch_dtype, device_map="auto"
            )
        if (text_encoder_path := checkpoint_path / "text_encoder").exists():
            kwargs["text_encoder"] = CLIPTextModel.from_pretrained(
                text_encoder_path, torch_dtype=self.config.torch_dtype, device_map="auto"
            )
        pipeline = StableDiffusionPipeline.from_pretrained(self.config.base_model_name_or_path, **kwargs)

        self.loaded_model = (checkpoint, pipeline)
        return pipeline

    async def load(self) -> None:
        """We don't load the model here because we don't know which checkpoint to load yet."""
        self.config = get_config(InferenceConfig)

    async def predict(self, examples: list[bytes]) -> dict[str, list[str]]:
        """Generate images from the given prompt and checkpoint.

        Args:
            examples: A list of bytes, where each byte is a JSON string containing the following keys:
                - checkpoint: The checkpoint to use for generation
                - prompt: The prompt to use for generation
                - **kwargs: Any additional keyword arguments to pass to StableDiffusionPipeline

        Returns:
            A dict with key "images" and value a list of base64-encoded PNG images.
        """
        input_json = json.loads(examples[0].decode("utf-8"))
        logger.info(f"Input params: {input_json}")

        if "checkpoints_available" in input_json:
            checkpoints = [p.name for p in self.config.checkpoints_path.iterdir() if "step_" in p.name]
            return {"checkpoints_available": checkpoints}

        images = self.generate_image(input_json.pop("checkpoint"), input_json.pop("prompt"), **input_json)
        b64_encoded_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            b64_encoded_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        return {"images": b64_encoded_images}


if __name__ == "__main__":
    model = Dreambooth()
    model.start()
