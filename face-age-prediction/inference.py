from __future__ import annotations

import io
import os
import time
import json
from base64 import b64decode
from pathlib import Path
from typing import Dict

import cv2
import torch
import numpy as np
from torch import nn
from PIL import Image
from slingshot import InferenceModel
import torchvision
from torchvision import transforms

FACE_DETECTION_MODEL_NAME = "res10_300x300_ssd_iter_140000.caffemodel"
FACE_DETECTION_CONFIG_NAME = "deploy.prototxt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = Path("/mnt/face-age-prediction-model/best_mae.pth")
NUM_AGE_CLASSES = 101


class FaceAgeDeployment(InferenceModel):
    face_detection_model_path = Path('/mnt/model/')

    def get_face_age_model(self) -> nn.Module:
        """The ResNet18 model pre-trained on ImageNet with a linear layer on top for age prediction."""
        self.face_age_model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.face_age_model.fc = nn.Linear(
            self.face_age_model.fc.in_features, NUM_AGE_CLASSES
        )
        self.face_age_model = self.face_age_model.to(DEVICE)

        face_age_checkpoint_path = os.path.join(CHECKPOINT_PATH)
        checkpoint: Dict[str, torch.Tensor] = torch.load(
            face_age_checkpoint_path, map_location=DEVICE
        )
        self.face_age_model.load_state_dict(checkpoint['state_dict'])
        self.face_age_model.eval()
        self.face_age_model = self.face_age_model.to(DEVICE)

    @torch.inference_mode()
    def _detect_face_and_crop_square(self, img, img_h, img_w) -> list[int]:
        """Detects a face in an image and crops it to a square."""
        # TODO: clean up this try except/check exception types explicitly for cv2
        try:
            # Optimal parameters based on https://github.com/opencv/opencv/tree/master/samples/dnn#face-detection
            blob = cv2.dnn.blobFromImage(
                cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 117.0, 123.0)
            )
            self.face_detection_net.setInput(blob)
            faces = self.face_detection_net.forward()
            if faces.shape[2] == 0:
                raise ValueError("No faces detected")

            # Crop original image to square bounding box of first detected face
            # [0, 0, 0] means we are looking at the first face in the first image of the first batch
            # and [3:7] means we are looking at the bounding box coordinates since the first 3 values
            # contain the class label, and confidence score
            x1, y1, x2, y2 = faces[0, 0, 0, 3:7] * np.array(
                [img_h, img_w, img_h, img_w]
            )
            x1, y1, x2, y2 = round(x1), round(y1), round(x2), round(y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            return [x1, y1, x2, y2, w, h]

        except Exception as e:
            print(f"Error detecting face: {e}")
            print(f"Cropping image to square without face detection...")
            min_square_dim = min(img.size)

            # Center crop square from image
            left = (img.size[0] - min_square_dim) // 2
            top = (img.size[1] - min_square_dim) // 2
            return [
                left,
                top,
                left + min_square_dim,
                top + min_square_dim,
                min_square_dim,
                min_square_dim,
            ]

    @torch.inference_mode()
    def _predict_age(
        self,
        img,
        img_size=224,
        margin=0.4,
    ) -> float:
        """Predicts the age of a person in an image."""

        transforms_aug_val = transforms.Compose([transforms.ToTensor()])

        img_w, img_h = img.shape[:2]
        face_rectangle = self._detect_face_and_crop_square(img, img_h, img_w)

        left, top, right, bottom, width, height = face_rectangle

        # extend the face rectangle to include a margin around the face
        # which includes some background, hair, and clothing of the subject
        left_extended = max(int(left - margin * width), 0)
        top_extended = max(int(top - margin * height), 0)
        right_extended = min(int(right + margin * width), img_w - 1)
        bottom_extended = min(int(bottom + margin * height), img_h - 1)
        cv2.rectangle(img, (left, top), (right, bottom), (255, 255, 255), 2)
        cv2.rectangle(
            img,
            (left_extended, top_extended),
            (right_extended, bottom_extended),
            (255, 0, 0),
            2,
        )

        # crop the face and resize it to the model's required input size
        inputs = cv2.resize(
            img[top_extended : bottom_extended + 1, left_extended : right_extended + 1],
            (img_size, img_size),
        )

        inputs = transforms_aug_val(inputs)

        inputs = inputs.unsqueeze(0).to(DEVICE)

        # Generates the probability distribution of the age. For example, if the face is of a 20-year-old,
        # the outputs will contain higher probabilities for ages around 20 and
        # very low values for ages far from 20.
        outputs = self.face_age_model(inputs)

        outputs = outputs.softmax(-1)

        ages = torch.arange(0, NUM_AGE_CLASSES).to('cuda').float()

        # Expected age calculation
        outputs = (outputs * ages).sum(axis=-1)

        return outputs.item()

    async def load(self) -> None:
        config_path = os.path.join(
            self.face_detection_model_path, FACE_DETECTION_CONFIG_NAME
        )
        detection_model_path = os.path.join(
            self.face_detection_model_path, FACE_DETECTION_MODEL_NAME
        )
        self.face_detection_net = cv2.dnn.readNetFromCaffe(
            config_path, detection_model_path
        )

        self.get_face_age_model()

    async def predict(self, examples: list[bytes]) -> dict[str | float]:
        examples = json.loads(examples[0])
        image_bytes = b64decode(examples["image"])

        img = Image.open(io.BytesIO(image_bytes))

        img = np.array(img)

        if img.shape[2] == 4:
            img = img[:, :, :3]

        start = time.monotonic()

        age = self._predict_age(img)

        end = time.monotonic()
        print(f"Prediction took {end - start} seconds on {DEVICE}")

        return {"age": age}


if __name__ == "__main__":
    model = FaceAgeDeployment()
    model.start()
