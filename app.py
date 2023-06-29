import io

import gradio as gr
from fastapi import FastAPI
from PIL import Image

from slingshot import SlingshotSDK

DEPLOYMENT_NAME = "classifier-deployment"

app = FastAPI()
sdk = SlingshotSDK()


async def predict_digit(img) -> int:  # TODO Type input (it's a numpy array)
    # Get the bytes from the image
    img_bytes = io.BytesIO()
    img = Image.fromarray(img)
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()
    resp = await sdk.predict(deployment_name=DEPLOYMENT_NAME, example_bytes=img_bytes)
    if "prediction" not in resp["data"]:
        raise Exception(f"Error running inference: {resp}")

    digit_pred = resp["prediction"]
    return digit_pred


demo = gr.Interface(fn=predict_digit, inputs="sketchpad", outputs="label")

app = gr.mount_gradio_app(app, demo, path="/")
