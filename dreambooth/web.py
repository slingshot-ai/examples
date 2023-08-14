import base64
import json
from io import BytesIO

import gradio as gr
from PIL import Image

from slingshot.sdk import SlingshotSDK

sdk = SlingshotSDK()
DEPLOYMENT_NAME = "image-generation"

IMG_SIZE = 512


async def generate_image(
    prompt: str, negative_prompt: str, num_inference_steps: int = 50, num_samples: int = 1, guidance_scale: float = 7
) -> Image:
    input_params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": IMG_SIZE,
        "width": IMG_SIZE,
        "num_samples": num_samples,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    input_bytes = json.dumps(input_params).encode("utf-8")
    resp = await sdk.predict(deployment_name=DEPLOYMENT_NAME, example_bytes=input_bytes)

    if "images" not in resp:
        raise Exception(f"Error running inference: {resp}")

    images = []
    for img_b64 in resp["images"]:
        img = Image.open(BytesIO(base64.b64decode(img_b64)))
        images.append(img)

    return images


demo = gr.Interface(
    fn=generate_image,
    inputs=[
        "text",
        "text",
        gr.Slider(1, 200, value=50, label="Num Denoising Steps", info="Number of steps to denoise the image"),
        gr.Slider(1, 5, value=2, label="Num Samples", info="Number of samples to generate"),
        gr.Slider(1, 200, value=7, label="Guidance Scale", info="Guidance scale"),
    ],
    outputs=gr.Gallery(label="Generated Images", allow_preview=True, show_download_button=True),
    title="Dreambooth",
    allow_flagging="never",
).queue(concurrency_count=1)

demo.launch(server_name="0.0.0.0", server_port=8080)
