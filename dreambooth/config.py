from __future__ import annotations

from pathlib import Path
from pydantic import BaseModel, Field


class DreamboothConfigTrain(BaseModel):
    # If the user sets resume_training as True but there are no pre-trained models available (check if there is a bin
    # file under pretrain_model_name_or_path path). So, the pretrain_model_name_or_path is now set to this such that
    # the base model can be picked.
    base_model_path: Path = Field(
        Path('runwayml/stable-diffusion-v1-5'), alias='BASE_MODEL_PATH'
    )

    # The name or unique identifier for the concept the model is learning.
    instance_name: str = Field(
        'DishaniL', alias='INSTANCE_NAME'
    )

    # Mounted directory with the images that the model is learning.
    instance_data_dir: Path = Path(
        '/mnt/dreambooth_input_data/instance_images'
    )

    # Whether to resume training from a checkpoint. If False, the model will be trained from scratch. If True, the
    # model will be trained from the last saved checkpoint if the base model is provided.
    resume_training: bool = Field(
        False, alias='RESUME_TRAINING'
    )

    unet_training_steps: int = Field(1800, alias='UNET_TRAINING_STEPS')
    unet_learning_rate: float = Field(3e-6, alias='UNET_LEARNING_RATE')
    text_encoder_training_steps: int = Field(350, alias='TEXT_ENCODER_TRAINING_STEPS')
    text_encoder_learning_rate: float = Field(1e-6, alias='TEXT_ENCODER_LEARNING_RATE')

    # Whether to train the text encoder. This is False when we are only training the UNet.
    train_text_encoder: bool = False

    save_text_encoder: bool = True

    # Whether to offset the noise with random values so that we can produce images with a higher dynamic range.
    offset_noise: bool = Field(
        False, alias='OFFSET_NOISE'
    )

    # Whether to use image captions for each image used for training. For our case, it is only the name of the
    # identifier. It is used in train_dreambooth.py of the original repo.
    image_captions_filename: bool = True

    train_only_unet: bool = False
    save_starting_step: int = 1
    save_n_steps: int = Field(500, alias='SAVE_CHECKPOINT_EVERY')

    # Model huggingface tag or path to weights on disk
    pretrained_model_name_or_path: Path = Field(
        Path('runwayml/stable-diffusion-v1-5'), alias='BASE_MODEL_PATH'
    )

    # The VAE model will be used from this repo's file.
    pretrained_vae_path: Path = Field(
        Path('stabilityai/sd-vae-ft-mse'), alias='VAE_MODEL_PATH'
    )

    # The output directory where the model checkpoints will be written. During re-training, if there is a model saved
    # here then it gets loaded from this path.
    output_dir: Path = Path(
        "/mnt/dreambooth-trained-model-example"
    )

    # Whether to use captions from a file for training images.
    external_caption: bool = Field(
        False, alias='EXTERNAL_CAPTIONS'
    )

    # A folder containing the captions of images in the training data.
    captions_dir: Path = Path(
        "/mnt/dreambooth_input_data/captions"
    )

    seed: int | str = Field('', alias='SEED')  # If provided, does not use a random seed.

    # The resolution of images that are expected by the model. Currently, testing has been done
    # on resolution 512 model itself and the SD 1.5 model used is fine-tuned on 512x512 resolution images.
    resolution: int = 512

    mixed_precision: str = Field("fp16", alias='PRECISION')  # This is needed if you are training on a 16GB GPU.

    # The directory where the model checkpoints have been saved using previous training rounds and the pretrained model
    # will be picked up from this mount.
    model_saved_output_dir: Path = Path(
        '/mnt/dreambooth-trained-model-example'
    )
    enable_text_encoder_training: bool = True


class TextToImageParams(BaseModel):
    # Please ensure that the prompt is very descriptive.
    prompt: str = Field(
        'closeup portrait of DishaniL woman with mouth open, highly detailed, science fiction, star wars concept art, '
        'intricate details, bright colors, golden hour, art by marko djurdjevic, greg rutkowski, wlop, fredperry, '
        'digital painting, rossdraws',
        alias='PROMPT',
    )

    # Negative prompts are also very useful to get the model to focus on the right things.
    negative_prompt: str = Field(
        'ugly, disfigured face', alias='NEGATIVE_PROMPT'
    )
    num_samples: int = Field(4, alias='NUM_SAMPLES')

    # The guidance scale controls how much importance is given to the textual prompt when generating an image.
    guidance_scale: int = Field(
        7, alias='GUIDANCE_SCALE'
    )

    num_inference_steps: int = Field(40, alias='NUM_INFERENCE_STEPS')
    height: int = Field(512, alias='HEIGHT')
    width: int = Field(512, alias='WIDTH')
