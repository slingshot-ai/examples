from __future__ import annotations
from pathlib import Path
import requests

# Define a directory to save the model and the URLs for the model files
MODEL_DIR = Path('/mnt/face_detector')
MODEL_FILE = {
    'path': "res10_300x300_ssd_iter_140000.caffemodel",
    'url': 'https://raw.githubusercontent.com/spmallick/learnopencv/blob/master/FaceDetectionComparison/models/res10_300x300_ssd_iter_140000_fp16.caffemodel',
}
CONFIG_FILE = {
    'path': "deploy.prototxt",
    'url': 'https://raw.githubusercontent.com/spmallick/learnopencv/master/FaceDetectionComparison/models/deploy.prototxt',
}


def download_file(file_info: dict) -> None:
    save_path = MODEL_DIR / file_info['path']
    print(f"Downloading {file_info['url']} to {save_path}...")
    response = requests.get(file_info['url'], stream=file_info['is_binary'])
    response.raise_for_status()

    with save_path.open('wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
        

def download_face_detection_model() -> None:
    """
    Downloads the face detection model and config from the OpenCV repo. The model is used to detect faces in images
    before passing them to the face age model at inference time.
    """
    print(f"Downloading face detection model to {MODEL_DIR}...")
    download_file(MODEL_FILE)
    print(f"Downloading face detection config to {MODEL_DIR}...")
    download_file(CONFIG_FILE)
    print("Done!")


def main():
    download_face_detection_model()


if __name__ == "__main__":
    main()
