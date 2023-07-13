from pathlib import Path

import torch

from slingshot import InferenceModel, Prediction
from utils import bytes_to_tensor


class MNISTInference(InferenceModel):
    async def load(self) -> None:
        """
        Slingshot will call this method to load the model.

        Implementation example:
            self.model = torch.load("/mnt/model/model.pt")
        """
        model_path = Path('/mnt/model/model.pt')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.hn_model.device}")
        self.model = torch.jit.load(str(model_path)).to(self.device)
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
        img_example = bytes_to_tensor(examples[0]).to(self.device)
        prob, index = self.model(img_example).squeeze().softmax(0).max(0)
        return {'confidence': prob.item(), 'prediction': str(index.item())}


if __name__ == "__main__":
    model = MNISTInference()
    model.start()