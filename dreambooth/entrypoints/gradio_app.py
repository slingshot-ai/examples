import base64
import json
import logging
from io import BytesIO

import gradio as gr
from PIL import Image
from slingshot.sdk import SlingshotSDK

DEPLOYMENT_NAME = "image-generation"

sdk = SlingshotSDK()
logger = logging.getLogger("dreambooth")


async def generate_image(
    checkpoint: str, prompt: str, num_inference_steps: int, num_samples: int, guidance_scale: float
) -> list[Image]:
    if checkpoint == "":
        raise gr.Error("Please select a model")
    if prompt == "":
        raise gr.Error("Please enter a prompt")

    input_params = {
        "checkpoint": checkpoint,
        "prompt": prompt,
        "num_images_per_prompt": num_samples,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
    }
    logger.info(f"Input params: {input_params}")

    input_bytes = json.dumps(input_params).encode("utf-8")
    resp = await sdk.predict(deployment_name=DEPLOYMENT_NAME, example_bytes=input_bytes, timeout_seconds=300)

    if "images" not in resp:
        raise gr.Error(f"Error running inference: {resp}")

    images = []
    for img_b64 in resp["images"]:
        img = Image.open(BytesIO(base64.b64decode(img_b64)))
        images.append(img)

    return images


async def get_model_choices():
    input_bytes = json.dumps({"checkpoints_available": "Anything"}).encode("utf-8")
    resp = await sdk.predict(deployment_name=DEPLOYMENT_NAME, example_bytes=input_bytes, timeout_seconds=30)
    checkpoint_choices = sorted(resp["checkpoints_available"])
    return gr.Dropdown.update(choices=checkpoint_choices)


def example_prompts() -> list[str]:
    with open("prompts.txt", "r") as f:
        return f.read().splitlines()


with gr.Blocks() as demo:
    with gr.Column():
        with gr.Row():
            title = gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Dreambooth</h1>")
        with gr.Row():
            with gr.Column():
                checkpoint = gr.Dropdown(choices=[], label="Model")  # choices populated in demo.load()
                prompt = gr.Dropdown(choices=example_prompts(), allow_custom_value=True, label="Prompt")
                num_inference_steps = gr.Slider(1, 200, value=50, step=1, label="Num inference steps")
                num_samples = gr.Slider(1, 8, value=1, step=1, label="Num samples")
                guidance_scale = gr.Slider(0, 20, value=7.5, label="Guidance scale")
                submit = gr.Button(value="Generate", variant="primary")
            with gr.Column():
                generated_images = gr.Gallery(label="Generated Images", allow_preview=True, show_download_button=True)

    demo.load(get_model_choices, outputs=checkpoint)
    submit.click(
        generate_image,
        inputs=[checkpoint, prompt, num_inference_steps, num_samples, guidance_scale],
        outputs=generated_images,
    )


demo.queue(max_size=1)
demo.launch(server_name="0.0.0.0", server_port=7860)
