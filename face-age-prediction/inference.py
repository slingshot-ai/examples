from __future__ import annotations

import io
import time
from pathlib import Path

import torch
import torchvision.transforms as transforms
from model import FaceAgeTrainer
from PIL import Image
from pydantic import BaseModel
from slingshot import InferenceModel
from slingshot.sdk.utils import get_config


class DeployConfig(BaseModel):
    # Age prediction model
    checkpoint_path: Path = Path("/mnt/face-age-prediction-model/model.ckpt")
    img_size: int = 224


class FaceAgeDeployment(InferenceModel):
    @torch.inference_mode()
    def _predict_age(self, img: Image) -> float:
        """Predicts the age of a person in an image."""
        processed_img = self.age_prediction_processor(img).unsqueeze(0).to(self.device)
        age = self.age_prediction_model(processed_img)
        return age.item()

    async def load(self) -> None:
        # Load config
        self.config = get_config(DeployConfig)

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Loading age prediction model...")
        self.age_prediction_model = FaceAgeTrainer.load_from_checkpoint(self.config.checkpoint_path).to(self.device)
        self.age_prediction_model.eval()
        self.age_prediction_processor = transforms.Compose(
            [transforms.Resize([self.config.img_size, self.config.img_size]), transforms.ToTensor()]
        )

    async def predict(self, examples: list[bytes]) -> dict[str | float]:
        # Load image
        image_bytes = examples[0]
        img = Image.open(io.BytesIO(image_bytes))

        # Predict age
        start = time.monotonic()
        age = self._predict_age(img)
        end = time.monotonic()
        print(f"Prediction took {end - start} seconds on {self.device}")

        return {"age": age}


if __name__ == "__main__":
    model = FaceAgeDeployment()
    model.start()
