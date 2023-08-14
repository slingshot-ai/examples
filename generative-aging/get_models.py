from __future__ import annotations

import os
import shutil
from pathlib import Path
import requests
from zipfile import ZipFile

import gdown

FACE_DETECTION_MODEL_PATH = (
    "/mnt/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)
FACE_DETECTION_CONFIG_PATH = "/mnt/face_detector/deploy.prototxt"

RESNET_MODEL_PATH = "/mnt/deeplab_model/R-101-GN-WS.pth.tar"
SEGMENTATION_MODEL_PATH = "/mnt/deeplab_model/deeplab_model.pth"

AGING_MALES_MODEL_PATH = "/mnt/checkpoints-male/males_model.zip"
AGING_FEMALES_MODEL_PATH = "/mnt/checkpoints-female/females_model.zip"


def get_face_detection_models():
    folder_path = Path('/mnt/face_detector')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading face detection model...")
    r = requests.get(
        'https://github.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel?raw=true'
    )
    with open(FACE_DETECTION_MODEL_PATH, 'wb') as f:
        f.write(r.content)

    print("Downloading face detection config...")
    r = requests.get(
        'https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt'
    )
    with open(FACE_DETECTION_CONFIG_PATH, 'w') as f:
        f.write(r.text)


def get_resnet_for_deeplabv3_segmentation():
    folder_path = Path('/mnt/deeplab_model')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    gdown.download(
        'https://drive.google.com/uc?id=1oRGgrI4KNdefbWVpw0rRkEP1gbJIRokM',
        RESNET_MODEL_PATH,
    )

    gdown.download(
        'https://drive.google.com/uc?id=1w2XjDywFr2NjuUWaLQDRktH7VwIfuNlY',
        SEGMENTATION_MODEL_PATH,
    )


def get_aging_models():
    folder_path = Path('/mnt/checkpoints')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    gdown.download(
        'https://drive.google.com/uc?id=1MsXN54hPi9PWDmn1HKdmKfv-J5hWYFVZ',
        AGING_MALES_MODEL_PATH,
    )

    gdown.download(
        'https://drive.google.com/uc?id=1LNm0zAuiY0CIJnI0lHTq1Ttcu9_M1NAJ',
        AGING_FEMALES_MODEL_PATH,
    )

    dest_path_male = Path("/mnt/checkpoints-male")
    dest_path_female = Path("/mnt/checkpoints-female")

    with ZipFile(AGING_MALES_MODEL_PATH, 'r') as zObject:
        zObject.extractall(path=str(dest_path_male))

    with ZipFile(AGING_FEMALES_MODEL_PATH, 'r') as zObject:
        zObject.extractall(path=str(dest_path_female))

    checkpoint_name = 'latest_net_g_running.pth'

    shutil.move(
        os.path.join(dest_path_male, "males_model", checkpoint_name), dest_path_male
    )

    shutil.move(
        os.path.join(dest_path_female, "females_model", checkpoint_name),
        dest_path_female,
    )


def main():
    get_face_detection_models()
    get_resnet_for_deeplabv3_segmentation()
    get_aging_models()


if __name__ == "__main__":
    main()
