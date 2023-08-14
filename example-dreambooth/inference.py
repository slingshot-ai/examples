import base64
import json
from io import BytesIO
from pathlib import Path

import torch
from config import TextToImageParams
from diffusers import DDIMScheduler, StableDiffusionPipeline
from PIL import Image
from torch import autocast

from slingshot import InferenceModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class Dreambooth(InferenceModel):
    pipe: StableDiffusionPipeline = None
    scheduler: DDIMScheduler = None
    model_path: Path = Path('/mnt/model')

    @torch.inference_mode()
    def generate_image(self, input_params: TextToImageParams) -> list[Image]:
        # Here, 'autocast' ensures mixed-precision computations and 'torch.inference_mode' ensures that the model does
        # not compute gradients.
        with autocast("cuda"):
            return self.pipe(
                input_params["prompt"],
                height=input_params["height"],
                width=input_params["width"],
                negative_prompt=input_params["negative_prompt"],
                num_images_per_prompt=input_params["num_samples"],
                num_inference_steps=input_params["num_inference_steps"],
                guidance_scale=input_params["guidance_scale"],
                generator=None,
            ).images

    async def load(self) -> None:
        self.scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )

        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path, scheduler=self.scheduler, safety_checker=None, torch_dtype=torch.float16
        ).to(DEVICE)

    async def predict(self, examples: list[bytes]) -> dict[str, list[str]]:
        """
        This method takes prompts, negative prompts, and other parameters as input and returns a list of Dreambooth generated images as base64 encoded images.
        """

        input_json = json.loads(examples[0].decode("utf-8"))

        images = self.generate_image(input_json)
        b64_encoded_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            b64_encoded_images.append(base64.b64encode(buffered.getvalue()).decode("utf-8"))
        return {"images": b64_encoded_images}


if __name__ == "__main__":
    model = Dreambooth()
    model.start()