import sys
import os

import cv2
import torch
from PIL import Image
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms

import deeplab

FACE_ID = 2
HAIR_ID = 11
BACKGROUND_COLOR = 127

FACE_DETECTION_MODEL_PATH = (
    "/mnt/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)
FACE_DETECTION_CONFIG_PATH = "/mnt/face_detector/deploy.prototxt"

SEGMENTATION_MODEL_NAME = '/mnt/deeplab_model/deeplab_model.pth'


def load_face_detection_model() -> nn.Module:
    """Loads the face detection model."""
    print("Loading face detection model")
    return cv2.dnn.readNetFromCaffe(
        FACE_DETECTION_CONFIG_PATH, FACE_DETECTION_MODEL_PATH
    )


def detect_face_and_crop_square(
    img: Image,
    face_detection_net: nn.Module,
    img_h: int,
    img_w: int,
    margin: float = 0.4,
) -> np.array:
    """Detects a face in an image and crops it to a square."""
    # Optimal parameters based on https://github.com/opencv/opencv/tree/master/samples/dnn#face-detection
    blob = cv2.dnn.blobFromImage(
        cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
    )
    face_detection_net.setInput(blob)
    faces = face_detection_net.forward()
    if faces.shape[2] == 0:
        raise ValueError("No faces detected")

    # Crop original image to square bounding box of first detected face
    # [0, 0, 0] means we are looking at the first face in the first image from the unscaled image level
    # and [3:7] means we are looking at the bounding box coordinates since the first 3 values
    # contain the class label, and confidence score
    left, top, right, bottom = faces[0, 0, 0, 3:7] * np.array(
        [img_h, img_w, img_h, img_w]
    )
    left, top, right, bottom = round(left), round(top), round(right), round(bottom)
    width, height = abs(right - left), abs(bottom - top)

    left_extended = max(int(left - margin * width), 0)
    top_extended = max(int(top - margin * height), 0)
    right_extended = min(int(right + margin * width), img_w - 1)
    bottom_extended = min(int(bottom + margin * height), img_h - 1)

    height_face = bottom_extended - top_extended
    width_face = right_extended - left_extended

    # If the face is not a square, extend the shorter side to make it a square. Faces are generally taller than they are wide, so we extend the width.

    if height_face > width_face:
        left_extended = max(int(left_extended - (height_face - width_face) / 2), 0)
        right_extended = min(
            int(right_extended + (height_face - width_face) / 2), img_w - 1
        )

    face = img[top_extended:bottom_extended, left_extended:right_extended]

    return top_extended, bottom_extended, left_extended, right_extended, face


def segment_face_hair(
    orig_img: np.array,
    deeplab_model: nn.Module,
    deeplab_input_size: int = 513,
    out_size: int = 256,
) -> np.array:
    """Segments the face and hair from an image using deeplabv3 model for segmentation."""
    img = cv2.resize(orig_img, (deeplab_input_size, deeplab_input_size))
    deeplab_data_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    img = deeplab_data_transform(img)
    img = img.cuda()
    deeplab_model.cuda()
    outputs = deeplab_model(img.unsqueeze(0))
    deeplab_model.cpu()

    _, pred = torch.max(outputs, 1)
    pred = pred.data.cpu().numpy().squeeze().astype(np.uint8)
    seg_map = Image.fromarray(pred)
    seg_map = np.uint8(seg_map.resize((out_size, out_size), Image.NEAREST))

    orig_img = cv2.resize(orig_img, (out_size, out_size))

    labels_to_mask = [0, 14, 15, 16, 18]
    # Mask out all labels except face components and hair
    for idx in labels_to_mask:
        orig_img[seg_map == idx] = BACKGROUND_COLOR

    return orig_img


def load_segmentation_model() -> nn.Module:
    """Loads the deeplabv3 model for segmentation."""
    deeplab_model = getattr(deeplab, 'resnet101')(
        pretrained=True,
        num_classes=19,
        num_groups=32,
        weight_std=True,
        beta=False,
    )

    deeplab_model.eval()

    checkpoint = torch.load(SEGMENTATION_MODEL_NAME)
    state_dict = {
        k[7:]: v for k, v in checkpoint['state_dict'].items() if 'tracked' not in k
    }
    deeplab_model.load_state_dict(state_dict)

    print("Loaded segmentation model")

    return deeplab_model


def main(img_name: str) -> None:
    image = cv2.imread(img_name)
    img_w, img_h = image.shape[:2]

    face_detection_net = load_face_detection_model()

    (
        top_extended,
        bottom_extended,
        left_extended,
        right_extended,
        cropped_face,
    ) = detect_face_and_crop_square(image, face_detection_net, img_h, img_w)

    cropped_face_height, cropped_face_width = abs(top_extended - bottom_extended), abs(
        left_extended - right_extended
    )

    deeplab_model = load_segmentation_model()

    segmented_face = segment_face_hair(cropped_face, deeplab_model)

    segmented_face = cv2.resize(
        segmented_face, (cropped_face_width, cropped_face_height)
    )
    kernel = np.ones((3, 3), np.uint8)
    mask = (segmented_face != BACKGROUND_COLOR).astype(np.uint8)

    mask = cv2.erode(mask, kernel, iterations=1)
    mask = mask.astype(bool)

    segmented_face = np.where(mask, segmented_face, cropped_face)

    image[top_extended:bottom_extended, left_extended:right_extended] = segmented_face

    cv2.imwrite(
        os.path.join(
            'results', img_name[:-4] + '_segmented_to_original_resolution.jpg'
        ),
        image,
    )


if __name__ == '__main__':
    img_name = sys.argv[1]
    main(img_name)
