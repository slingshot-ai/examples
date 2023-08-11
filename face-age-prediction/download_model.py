from __future__ import annotations

import os
from pathlib import Path
import requests

FACE_DETECTION_MODEL_PATH = (
    "/mnt/face_detector/res10_300x300_ssd_iter_140000.caffemodel"
)
FACE_DETECTION_CONFIG_PATH = "/mnt/face_detector/deploy.prototxt"


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


def main():
    get_face_detection_models()


if __name__ == "__main__":
    main()
