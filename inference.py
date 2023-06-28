from __future__ import annotations

from base64 import b64decode
from pathlib import Path
from typing import Any

import kserve
import torch
from utils import bytes_to_tensor


class InferenceModel(kserve.Model):
    model_path = Path('/mnt/model')

    def __init__(self) -> None:
        super().__init__("slingshot-model")  # Required by Slingshot
        self.model: Any | None = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(str(self.model_path / "model.pt")).to(self.device)
        self.ready = True

    async def predict(self, payload: dict[str, str], headers: dict[str, str] = None) -> dict[str, str | float]:
        """
        :param payload: the request payload from the client, where the example is base64 encoded as `example` with UTF-8
                        and any extra arguments are passed in as `extra_args`
        :param headers: the request headers from the client
        """
        img_example = b64decode(payload["example"].encode("utf-8"))
        img_example = bytes_to_tensor(img_example).unsqueeze(0).to(self.device)
        prob, index = self.model(img_example).squeeze().softmax(0).max(0)
        return {'confidence': prob.item(), 'prediction': str(index.item())}


if __name__ == "__main__":
    model = InferenceModel()
    kserve.ModelServer().start([model])
