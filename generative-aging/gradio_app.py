import io
import json
from base64 import b64encode, b64decode

import gradio as gr
from fastapi import FastAPI
from PIL import Image

from slingshot import SlingshotSDK

DEPLOYMENT_NAMES = {
    "male": "face_aging_deployment_male",
    "female": "face_aging_deployment_female",
}

app = FastAPI()
sdk = SlingshotSDK()


async def inference(img: Image, age: int, model_mode: str) -> Image:
    # Get the bytes from the image
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='PNG')
    img_bytes = img_bytes.getvalue()

    b64_img = b64encode(img_bytes).decode("utf-8")

    json_example = {"image": b64_img, "age": age}

    # convert json_example to bytes
    json_example = json.dumps(json_example).encode("utf-8")

    resp = await sdk.predict(
        deployment_name=DEPLOYMENT_NAMES[model_mode], example_bytes=json_example
    )

    if "aged" not in resp:
        raise Exception(f"Error running inference: {resp}")

    out_bytes = b64decode(resp["aged"])

    out_img = Image.open(io.BytesIO(out_bytes))

    return out_img


demo = gr.Interface(
    fn=inference,
    inputs=[
        gr.inputs.Image(type="pil"),
        gr.Slider(
            1,
            100,
            value=20,
            label="Desired Age",
            info="The desired Age for the output image",
        ),
        gr.Radio(
            ["male", "female"],
            label="Aging model to use",
            info="Which aging model to use",
        ),
    ],
    outputs=gr.outputs.Image(type="pil"),
)

app = gr.mount_gradio_app(app, demo, path="/")
