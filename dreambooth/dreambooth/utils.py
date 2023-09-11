import logging.config
from typing import Iterator, TypeVar

from torch.utils.data import DataLoader
import numpy as np


def setup_logging():
    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "standard": {
                    "format": "%(asctime)s [%(levelname)s] %(filename)s:%(lineno)d %(message)s",
                    "datefmt": "%m-%d %H:%M:%S",
                }
            },
            "handlers": {"default": {"level": "INFO", "formatter": "standard", "class": "logging.StreamHandler"}},
            "loggers": {"dreambooth": {"handlers": ["default"], "level": "INFO", "propagate": False}},
        }
    )


def tile_images(images: list[np.ndarray]) -> np.ndarray:
    """Tiles images into a single image.

    Args:
        images: A list of images to tile. All images should have 3 dimensions and the same shape.

    Returns:
        An approximately square image containing the images.
    """
    assert len(images) > 0
    height, width, channels = images[0].shape
    assert all(image.shape == (height, width, channels) for image in images), "All images must have the same shape."

    num_images = len(images)
    num_cols = int(np.ceil(np.sqrt(num_images)))
    num_rows = int(np.ceil(num_images / num_cols))

    tiled_image = np.zeros((height * num_rows, width * num_cols, channels), dtype=images[0].dtype)

    for i, image in enumerate(images):
        row = i // num_cols
        col = i % num_cols
        tiled_image[row * height : (row + 1) * height, col * width : (col + 1) * width] = image

    return tiled_image


T = TypeVar("T")


def infinite_loader(loader: DataLoader[T]) -> Iterator[T]:
    """Create an infinite stream of items from a torch DataLoader.

    Re-shuffles on each outer loop (if loader has shuffle=True)
    """
    while True:
        for x in loader:
            yield x
