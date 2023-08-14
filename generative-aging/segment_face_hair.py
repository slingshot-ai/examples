import sys

from transformers import AutoFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import torch.nn as nn
import numpy as np

FACE_ID = 2
HAIR_ID = 11


def main(img_name: str):
    extractor = AutoFeatureExtractor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = SegformerForSemanticSegmentation.from_pretrained(
        "mattmdjaga/segformer_b2_clothes"
    )

    image = Image.open(img_name)

    inputs = extractor(images=image, return_tensors="pt")

    inputs = inputs.to(model.device)

    outputs = model(**inputs)
    logits = outputs.logits.cpu()

    upsampled_logits = nn.functional.interpolate(
        logits,
        size=image.size[::-1],
        mode="bilinear",
        align_corners=False,
    )

    pred_seg = upsampled_logits.argmax(dim=1)[0]

    bg_mask = (pred_seg != FACE_ID) & (pred_seg != HAIR_ID)

    image_np = np.array(image)
    image_np[bg_mask] = 128  # make the background grey

    im = Image.fromarray(image_np)
    newsize = (256, 256)
    im = im.resize(
        newsize
    )  # resize to 256x256 to match the masked image size produced originally in the repo

    im.save(img_name.split('.')[0] + "_segmented.jpeg")


if __name__ == '__main__':
    img_name = sys.args[1]
    main(img_name)
