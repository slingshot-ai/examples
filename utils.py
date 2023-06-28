import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import torch
from pydantic import BaseModel


class MNISTExample(BaseModel):
    id: str
    example: str  # b64 encoded
    label: str


def base64_to_tensor(img_data: str):
    img_bytes = base64.b64decode(img_data.encode("utf-8"))
    return bytes_to_tensor(img_bytes)


def bytes_to_tensor(img_data: bytes):
    with BytesIO(img_data) as b:
        pil_image = Image.open(b).copy()

    pil_image = ImageOps.grayscale(pil_image)
    pil_image = pil_image.resize((28, 28))

    np_image = np.array(pil_image)
    im_arr = np_image.reshape((1, 28, 28))
    return torch.FloatTensor(im_arr / 255)
