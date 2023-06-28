import gradio as gr
from fastapi import FastAPI
from PIL import Image

from slingshot import SlingshotSDK

PROJECT_NAME = "example-mnist"
DEPLOYMENT_NAME = "classifier-deployment"

app = FastAPI()
sdk = SlingshotSDK()


async def predict_digit(img: Image) -> int:
    # Get the bytes from the image
    img_bytes = img.tobytes()
    await sdk.use_project(PROJECT_NAME)
    resp = await sdk.predict(deployment_name=DEPLOYMENT_NAME, example_bytes=img_bytes)
    if "data" not in resp or "prediction" not in resp["data"]:
        raise Exception(f"Error running inference: {resp}")

    digit_pred = resp["data"]["prediction"]
    return digit_pred


demo = gr.Interface(fn=predict_digit, inputs="sketchpad", outputs="label")

app = gr.mount_gradio_app(app, demo, path="/")
