from pathlib import Path
from typing import Iterator, Any

import torch.utils.data
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import CenterCrop, Compose, InterpolationMode, Normalize, RandomCrop, Resize, ToTensor
from transformers import PreTrainedTokenizer
from accelerate import Accelerator

from dreambooth.utils import infinite_loader
from dreambooth.config import TrainConfig


def get_entity_loader(
    image_dir: Path, caption: str, config: TrainConfig, tokenizer: PreTrainedTokenizer, accelerator: Accelerator
) -> Iterator[dict[str, torch.Tensor]]:
    """Create an infinite stream of batched image-caption_ids pairs.

    Call `next()` on the returned iterator to get the next batch.

    Right now, the dataset consists of images of the entity (loaded from `image_dir`) all with a fixed `caption`.
    To get fine-grained control over the image-caption pairs, you can define your own version of `ImageCaptionDataset`.
    """
    dataset = ImageCaptionDataset(image_dir, caption, config, tokenizer)
    collator = DataCollator(tokenizer.pad_token_id)
    loader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, collate_fn=collator)
    loader = accelerator.prepare(loader)
    return infinite_loader(loader)


class ImageCaptionDataset(Dataset):
    def __init__(self, image_dir: Path, caption: str, config: TrainConfig, tokenizer: PreTrainedTokenizer):
        assert image_dir.is_dir(), f"{image_dir} is not a directory"

        self.tokenizer = tokenizer

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
        return {
            "image": self.transform(item["image"]),
            "caption_ids": self.tokenizer(item["caption"], return_tensors="pt")["input_ids"][0],
        }


class DataCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        """Collects a list of image-caption_ids pairs and pads them into a batch.

        Args:
            examples: A list of image-caption_ids pairs. Each pair is a dict with keys "image" and "caption_ids".

        Returns:
            A dict with keys "input_ids", "attention_mask", and "pixel_values".
        """
        input_ids_tensor = [torch.tensor(example["caption_ids"]) for example in examples]
        input_ids = self.pad(input_ids_tensor, padding_value=self.pad_token_id)
        attention_mask = self.pad([torch.ones_like(t) for t in input_ids_tensor], padding_value=0)
        pixel_values = torch.stack([torch.tensor(example["image"]) for example in examples]).float()

        return {"input_ids": input_ids, "attention_mask": attention_mask, "pixel_values": pixel_values}

    @staticmethod
    def pad(tensors: list[torch.Tensor], padding_value: float) -> torch.Tensor:
        nested = torch.nested.nested_tensor(tensors)
        return torch.nested.to_padded_tensor(nested, padding=padding_value)
