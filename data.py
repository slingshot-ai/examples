import json
from base64 import b64encode
from io import BytesIO

import torchvision
from PIL import Image
from utils import MNISTExample

NUM_SAMPLES = 10000
OUTPUT_PATH = "/mnt/output/dataset.jsonl"
CACHE_FOLDER = "~/.cache/mnist"


def pil_image_to_base64(img: Image) -> str:
    with BytesIO() as b:
        img.save(b, "png")
        return b64encode(b.getvalue()).decode("utf-8")


def run():
    print("Downloading the dataset")
    mnist_train = torchvision.datasets.MNIST(CACHE_FOLDER, download=True, train=True)
    training_samples = [
        MNISTExample(id=str(i), example=pil_image_to_base64(img), label=str(digit))
        for i, (img, digit) in enumerate(mnist_train)
    ][:NUM_SAMPLES]

    print(f"Writing to {OUTPUT_PATH}")
    with open(OUTPUT_PATH, "w") as f:
        for sample in training_samples:
            f.write(json.dumps(sample.dict()) + "\n")
    print("Complete")


if __name__ == '__main__':
    run()
