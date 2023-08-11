import base64
import json
from io import BytesIO

import gradio as gr
from fastapi import FastAPI
from PIL import Image

from slingshot.sdk import SlingshotSDK

app = FastAPI()
sdk = SlingshotSDK()

DEPLOYMENT_NAME = "image-generation"  # deployment name must match the deployment name in our Slingshot project


async def generate_image(
    prompt: str,
    negative_prompt: str,
    height: int = 512,
    width: int = 512,
    num_inference_steps: int = 50,
    guidance_scale: float = 7,
) -> Image:
    input_params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_samples": 1,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    input_bytes = json.dumps(input_params).encode("utf-8")
    resp = await sdk.predict(deployment_name=DEPLOYMENT_NAME, example_bytes=input_bytes)
    if "data" not in resp or "images" not in resp["data"]:
        raise Exception(f"Error running inference: {resp}")

    img_b64 = resp["data"]["images"][0]
    img = Image.open(BytesIO(base64.b64decode(img_b64)))
    return img


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        "text",
        "text",
        gr.Slider(64, 1024, value=512, label="Height", info="Image height"),
        gr.Slider(64, 1024, value=512, label="Width", info="Image width"),
        gr.Slider(1, 200, value=50, label="Num Denoising Steps", info="Number of steps to denoise the image"),
        gr.Slider(1, 200, value=7, label="Guidance Scale", info="Guidance scale"),
    ],
    outputs="image",
)
app = gr.mount_gradio_app(app, demo, path="/")
