import base64
from io import BytesIO

from PIL import Image, ImageOps
import torch
from torchvision.transforms.functional import pil_to_tensor


def base64_to_tensor(img_data: str):
    img_bytes = base64.b64decode(img_data.encode("utf-8"))
    return bytes_to_tensor(img_bytes)


def bytes_to_tensor(img_data: bytes) -> torch.Tensor:
    """Takes in bytes from an image format recognized by PIL and returns a normalized 28x28 tensor."""
    pil_image = Image.open(BytesIO(img_data))
    return preprocess_pil_image(pil_image)


def preprocess_pil_image(pil_image: Image.Image) -> torch.Tensor:
    """Takes in a PIL image and returns a normalized 28x28 tensor."""
    pil_image = ImageOps.grayscale(pil_image)
    pil_image = pil_image.resize((28, 28))
    return pil_to_tensor(pil_image).to(float) / 255  # Returns shape (1, 28, 28)
