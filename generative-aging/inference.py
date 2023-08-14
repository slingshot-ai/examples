import time
import io
import json
from base64 import b64decode, b64encode
from fastapi import FastAPI

import cv2
import numpy as np
from PIL import Image

import data_loader
from slingshot import InferenceModel
from model_create import create_model
from preprocess import (
    load_face_detection_model,
    load_segmentation_model,
    detect_face_and_crop_square,
    segment_face_hair,
    BACKGROUND_COLOR,
)

app = FastAPI()


class FaceAgingDeployment(InferenceModel):
    async def load(self) -> None:
        self.model = create_model()
        self.face_detection_net = load_face_detection_model()
        self.deeplab_model = load_segmentation_model()

    async def predict(self, examples: list[bytes]) -> dict[str, str]:
        examples = json.loads(examples[0])

        image_bytes = b64decode(examples["image"])

        img = Image.open(io.BytesIO(image_bytes))

        age = examples["age"]

        output = self.inference(img, age)

        return output

    def preprocess(self, img: Image) -> Image:
        """Preprocess the image for inference by segmenting out the background and keeping the face and hair and keeping only the first detected face."""
        img_w, img_h = img.size[:2]

        img = np.array(img)

        (
            top_extended,
            bottom_extended,
            left_extended,
            right_extended,
            cropped_face,
        ) = detect_face_and_crop_square(img, self.face_detection_net, img_h, img_w)

        cropped_face_height = abs(top_extended - bottom_extended)

        cropped_face_width = abs(left_extended - right_extended)

        segmented_face = segment_face_hair(cropped_face, self.deeplab_model)

        return (
            top_extended,
            bottom_extended,
            left_extended,
            right_extended,
            cropped_face_height,
            cropped_face_width,
            cropped_face,
            Image.fromarray(segmented_face),
        )

    def inference(self, img: Image, age: int = 80) -> dict[str, str]:
        start = time.time()
        (
            top_extended,
            bottom_extended,
            left_extended,
            right_extended,
            cropped_face_height,
            cropped_face_width,
            cropped_face,
            preprocessed_img,
        ) = self.preprocess(img)

        data = data_loader.transform_image(preprocessed_img)

        out = self.model.inference(data, age)
        end = time.time()
        print("Inference computed in {} seconds".format(end - start))

        # scale by which we should downscale the cropped face and the coordinates of the cropped out face to the model output dimensions
        scale = out.shape[0] / cropped_face_height

        face_mask = np.ones(out.shape[:2], dtype=np.uint8)
        pixels_to_replace = np.all(
            out == [BACKGROUND_COLOR, BACKGROUND_COLOR, BACKGROUND_COLOR], axis=2
        )

        # mask has 1s where face is and 0s elsewhere (background)
        face_mask[pixels_to_replace] = 0

        # erode the mask in order to reduce the mask size and remove boundary artifacts
        kernel = np.ones((3, 3), np.uint8)
        face_mask = cv2.erode(face_mask, kernel, iterations=1)

        face_mask = face_mask != 0

        # resize the cropped face to the size of the output so that we can replace the background from the cropped face to the output
        cropped_face = cv2.resize(
            (np.array(cropped_face)), (out.shape[1], out.shape[0])
        )

        # Creates 3 copies of the mask, one for each color channel
        face_mask_np = np.tile(np.expand_dims(face_mask, -1), (1, 1, 3))

        # replace background with original background
        out_masked = np.where(face_mask_np, out, cropped_face)

        image = np.array(img)

        # downscale image such that the resized face fits
        new_image_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))

        image = cv2.resize(image, new_image_size, interpolation=cv2.INTER_CUBIC)

        top_extended, bottom_extended, left_extended, right_extended = (
            int(top_extended * scale),
            int(bottom_extended * scale),
            int(left_extended * scale),
            int(right_extended * scale),
        )

        # replace the aged face with appropriate background in the original image
        image[top_extended:bottom_extended, left_extended:right_extended] = out_masked[
            0 : bottom_extended - top_extended, 0 : right_extended - left_extended
        ]

        # encode numpy array as bytes
        output_bytes = io.BytesIO()
        output = Image.fromarray(image)
        output.save(output_bytes, format='PNG')

        output_bytes = output_bytes.getvalue()

        output_base64 = b64encode(output_bytes).decode("utf-8")

        return {"aged": output_base64}


if __name__ == "__main__":
    model = FaceAgingDeployment()
    model.start()
