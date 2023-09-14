import io

import gradio as gr
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel
from slingshot import SlingshotSDK
from slingshot.sdk.utils import get_config

app = FastAPI()
sdk = SlingshotSDK()


class DeployConfig(BaseModel):
    deployment_name: str = "age-prediction-deployment"


async def inference(img: Image) -> float:
    config = get_config(DeployConfig)
    deployment_name = config.deployment_name

    # Get the bytes from the image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="PNG")
    img_bytes = img_bytes.getvalue()

    resp = await sdk.predict(deployment_name=deployment_name, example_bytes=img_bytes)

    if "age" not in resp:
        raise gr.Error(f"Error running inference: {resp}")

    out_age = resp["age"]

    return out_age


demo = gr.Interface(fn=inference, inputs=gr.inputs.Image(type="pil"), outputs=gr.outputs.Textbox(label="Predicted Age"))

app = gr.mount_gradio_app(app, demo, path="/")
