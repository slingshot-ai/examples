import base64
import json
from io import BytesIO
from pathlib import Path

import torch
import uvicorn
from config import TextToImageParams
from diffusers import DDIMScheduler, StableDiffusionPipeline
from fastapi import FastAPI, Request
from PIL import Image
from torch import autocast

app = FastAPI()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model_is_ready = False
scheduler = None
pipe = None


@app.on_event("startup")
async def startup_event():
    global model_is_ready
    global pipe
    global scheduler

    scheduler = DDIMScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    model_path = Path('/mnt/model')

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path, scheduler=scheduler, safety_checker=None, torch_dtype=torch.float16
    ).to(DEVICE)
    model_is_ready = True


@torch.inference_mode()
def generate_image(input_params: TextToImageParams) -> list[Image]:
    # Here, 'autocast' ensures mixed-precision computations and 'torch.inference_mode' ensures that the model does not compute gradients.
    with autocast("cuda"), torch.inference_mode():
        return pipe(
            input_params.prompt,
            height=input_params.height,
            width=input_params.width,
            negative_prompt=input_params.negative_prompt,
            num_images_per_prompt=input_params.num_samples,
            num_inference_steps=input_params.num_inference_steps,
            guidance_scale=input_params.guidance_scale,
            generator=None,
        ).images


# V1 endpoints
# https://github.com/kserve/kserve/blob/c5f8984d3151769698664c33f94412b55a12d210/python/kserve/kserve/protocol/rest/server.py#L59


@app.get("/")
async def v1_liveness_check():
    return {"status": "alive"}


@app.get("/v1/models")
async def v1_models():
    return {"models": ["slingshot-model"]}


@app.get("/v1/models/{model_name}")
async def v1_model_metadata(model_name: str):
    return {"name": "slingshot-model", "ready": model_is_ready}


@app.post("/v1/models/slingshot-model:predict", response_model=None)
async def v1_predict(request: Request):
    input_dict: bytes = await request.body()
    input_data = json.loads(input_dict)
    txt_to_img_params: TextToImageParams = TextToImageParams(**input_data)
    images = generate_image(txt_to_img_params)
    b64_encoded_images = []
    for img in images:
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        b64_encoded_images.append(base64.b64encode(buffered.getvalue()))
    return {"images": b64_encoded_images}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
