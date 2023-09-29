import logging.config
from typing import Iterator, TypeVar

from torch.utils.data import DataLoader


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


T = TypeVar("T")


def infinite_loader(loader: DataLoader[T]) -> Iterator[T]:
    """Create an infinite stream of items from a torch DataLoader.

    Re-shuffles on each outer loop (if loader has shuffle=True)
    """
    while True:
        for x in loader:
            yield x
