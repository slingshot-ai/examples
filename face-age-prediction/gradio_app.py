import io
import json
import os
from base64 import b64encode

import gradio as gr
from fastapi import FastAPI
from PIL import Image
from pydantic import BaseModel

from slingshot import SlingshotSDK

app = FastAPI()
sdk = SlingshotSDK()


class DeployConfig(BaseModel):
    deployment_name: str = "face-age-predict"


async def inference(img: Image) -> Image:
    config = DeployConfig.parse_obj(json.loads(os.environ.get("CONFIG", "{}")))
    deployment_name = config.deployment_name

    # Get the bytes from the image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    b64_img = b64encode(img_bytes).decode("utf-8")

    json_example = {"image": b64_img} # TODO: Update to just sent the img_bytes

    # convert json_example to bytes
    json_example = json.dumps(json_example).encode("utf-8")

    resp = await sdk.predict(deployment_name=deployment_name, example_bytes=json_example)

    if "age" not in resp:
        raise Exception(f"Error running inference: {resp}")

    out_age = resp["age"]

    return out_age


demo = gr.Interface(
    fn=inference, inputs=[gr.inputs.Image(type="pil")], outputs=gr.outputs.Textbox(label="Predicted Age")
)

app = gr.mount_gradio_app(app, demo, path="/")
