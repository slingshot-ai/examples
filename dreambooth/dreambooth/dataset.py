from pathlib import Path
from typing import Iterator, Any

import torch.utils.data
from pillow_heif import register_heif_opener
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, RandomCrop, Resize, ToTensor
from transformers import PreTrainedTokenizer
from accelerate import Accelerator

from dreambooth.utils import infinite_loader
from dreambooth.config import TrainConfig

register_heif_opener()  # Allows us to load HEIC images (common on iPhones)


def get_entity_loader(
    image_dir: Path, caption: str, config: TrainConfig, tokenizer: PreTrainedTokenizer, accelerator: Accelerator
) -> Iterator[dict[str, torch.Tensor]]:
    """Create an infinite stream of batched image-caption_ids pairs.

    Call `next()` on the returned iterator to get the next batch.

    Right now, the dataset consists of images of the entity (loaded from `image_dir`) all with a fixed `caption`.
    To get fine-grained control over the image-caption pairs, you can define your own version of `ImageCaptionDataset`.
    """
    dataset = ImageCaptionDataset(image_dir, caption, config)
    collator = DataCollator(tokenizer)
    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collator)
    loader = accelerator.prepare(loader)
    return infinite_loader(loader)


class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir: Path, caption: str, config: TrainConfig):
        assert image_dir.is_dir(), f"{image_dir} is not a directory"

        self.data = []
        for image_path in image_dir.iterdir():
            if image_path.suffix.lower() in config.image_extensions.split(","):
                image = Image.open(image_path)
                if not image.mode == "RGB":
                    image = image.convert("RGB")
                self.data.append({"image": image, "caption": caption})

        assert len(self.data) > 0, f"Could not find any images in {image_dir}"

        self.transform = Compose(
            [
                Resize(config.resolution, interpolation=InterpolationMode.BILINEAR),
                CenterCrop(config.resolution) if config.center_crop else RandomCrop(config.resolution),
                ToTensor(),
                Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {"image": self.transform(item["image"]), "caption": item["caption"]}


class DataCollator:
    def __init__(self, tokenizer: PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collects a list of image-caption_ids pairs and pads them into a batch.

        Args:
            examples: A list of image-caption_ids pairs. Each pair is a dict with keys "image" and "caption_ids".

        Returns:
            A dict with keys "input_ids", "attention_mask", and "pixel_values".
        """
        captions = [example["caption"] for example in examples]
        inputs = self.tokenizer(captions, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = torch.stack([example["image"] for example in examples])

        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}
