import torch.utils.data
from PIL import Image

import torchvision.transforms as transforms


def get_transform(normalize: bool = True, output_size: int = 256) -> transforms.Compose:
    transform_list = []

    img_size = [output_size, output_size]
    # TODO: Segment hair and face, and crop the image. Use segment_face_hair.py as a reference.
    transform_list.append(transforms.Resize(img_size, interpolation=Image.NEAREST))

    transform_list.append(transforms.ToTensor())

    if normalize:
        mean = (0.5,)
        std = (0.5,)
        transform_list += [transforms.Normalize(mean, std)]

    return transforms.Compose(transform_list)


def transform_image(img: Image) -> dict:
    transform = get_transform()
    img = transform(img)
    img = img[:3]
    img = img.unsqueeze(0)

    return {
        'Imgs': img,
        'Paths': [''],
        'Classes': torch.zeros(1, dtype=torch.int),
        'Valid': True,
    }
