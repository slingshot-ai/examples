from pathlib import Path

import torch

from slingshot import InferenceModel, Prediction

from model import DigitRecognizer
from utils import bytes_to_tensor
import os


class MnistInference(InferenceModel):
    async def load(self) -> None:
        """
        Slingshot will call this method to load the model.

        Implementation example:
            self.model = torch.load("/mnt/model/model.pt")
        """
        model_path = Path("/mnt/model/model.ckpt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        print("Files in model path:", os.listdir("/mnt/model/"))
        # Check if file exists
        print("model file exists", model_path.exists())

        self.model = DigitRecognizer.load_from_checkpoint(model_path)
        print("model device", self.model.device)
        print("Model loaded")

    async def predict(self, examples: list[bytes]) -> Prediction | list[Prediction]:
        """
        Slingshot will call this method to make predictions, passing in the raw request bytes and returns a dictionary.
        For text inputs, the bytes will be the UTF-8 encoded string.

        If the model is not batched, the input will be a list with a single element and the output should be a single
        dictionary as the prediction response. Otherwise, the input will be a list of examples and the output should be
        a list of dictionaries with the same length and order as the input.

        Implementation example:
            example_text = examples[0].decode("utf-8")
            return self.model(example_text)
        """
        img_example = bytes_to_tensor(examples[0]).unsqueeze(0).to(self.device).to(torch.float32)
        print("image shape", img_example.shape)
        print("image type", img_example.dtype)
        print("model type", self.model)
        prob, index = self.model(img_example).squeeze().softmax(0).max(0)
        return {'confidence': prob.item(), 'prediction': str(index.item())}


if __name__ == "__main__":
    model = MnistInference()
    model.start()