# Code modified from "https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py"
from contextlib import nullcontext

import bitsandbytes as bnb
import torch
import torch.nn.functional as F
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from slingshot.sdk.utils import get_config
from transformers import CLIPTextModel, CLIPTokenizer

from dreambooth.config import TrainConfig
from dreambooth.dataset import get_entity_loader
from dreambooth.utils import tile_images, setup_logging


def main():
    config = get_config(TrainConfig)

    # --- Accelerator and logging setup ---
    config.logging_dir.parent.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        project_dir=config.logging_dir,
        log_with="wandb",
    )
    set_seed(config.seed)
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name=config.wandb_project,
            config=config.model_dump(),
            init_kwargs={"wandb": {"mode": "online" if not config.dry_run else "disabled"}},
        )

    setup_logging()
    logger = get_logger("dreambooth", log_level="INFO")
    logger.info("Config: %s", config.model_dump_json(indent=4))

    # --- Model and dataloader setup ---
    tokenizer = load_tokenizer(config)
    vae = load_vae(config)
    text_encoder = load_text_encoder(config)
    unet = load_unet(config)

    trainable_params = [
        *(p for p in vae.parameters() if p.requires_grad),
        *(p for p in text_encoder.parameters() if p.requires_grad),
        *(p for p in unet.parameters() if p.requires_grad),
    ]  # To initialize the optimizer and for grad norm clipping
    optimizer = load_optimizer(config, trainable_params)
    lr_scheduler = get_scheduler(
        config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=config.lr_warmup_steps * config.gradient_accumulation_steps,
        num_training_steps=config.max_train_steps * config.gradient_accumulation_steps,
    )
    vae, text_encoder, unet, optimizer, lr_scheduler = accelerator.prepare(
        vae, text_encoder, unet, optimizer, lr_scheduler
    )
    noise_scheduler = DDPMScheduler.from_pretrained(config.base_model_name_or_path, subfolder="scheduler")
    target_entity_loader = get_entity_loader(
        config.target_entity_image_dir, config.target_prompt, config, tokenizer, accelerator
    )
    if config.use_prior_preservation_loss:  # use prior preservation
        generic_entity_loader = get_entity_loader(
            config.generic_entity_image_dir, config.generic_prompt, config, tokenizer, accelerator
        )
    else:
        generic_entity_loader = None

    def compute_loss(batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # `batch` is the output of `DataCollator`
        latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
        latents = latents * vae.config.scaling_factor

        # Forward diffusion (add noise to the latents)
        noise = torch.randn_like(latents)
        batch_size = latents.shape[0]
        time_steps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=latents.device)
        noisy_latents = noise_scheduler.add_noise(latents, noise, time_steps)

        # Backward diffusion (denoise the latents)
        encoder_hidden_states = text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])[0]
        noise_pred = unet(noisy_latents, time_steps, encoder_hidden_states).sample
        return F.mse_loss(noise_pred, noise)

    # --- Training loop ---
    global_step = 1
    while global_step < config.max_train_steps:
        # --- Forward & backward pass ---
        with (
            accelerator.accumulate(unet) if config.train_unet else nullcontext(),
            accelerator.accumulate(text_encoder) if config.train_text_encoder else nullcontext(),
            accelerator.autocast(),
        ):
            loss = compute_loss(next(target_entity_loader))
            accelerator.backward(loss)
            total_loss = loss.detach()
            if config.use_prior_preservation_loss:
                # Doing forward and backward pass for the generic images separately to prevent going OOM
                prior_loss = compute_loss(next(generic_entity_loader)) * config.prior_preservation_weight
                accelerator.backward(prior_loss)
                total_loss += prior_loss.detach()
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, config.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        # --- Logging, saving, and evaluation ---
        accelerator.log({"loss": total_loss.item(), "lr": lr_scheduler.get_last_lr()[0]}, step=global_step + 1)
        if accelerator.sync_gradients:  # made a gradient step
            global_step += 1

            if accelerator.is_main_process:
                if global_step % config.save_n_steps == 0:
                    logger.info("Step %d: Saving model", global_step)
                    save_model(config, text_encoder, unet, global_step, accelerator)

                if global_step % config.eval_n_steps == 0:
                    logger.info("Generating samples at step %d", global_step)
                    image = generate_samples(vae, text_encoder, tokenizer, unet, noise_scheduler, config, accelerator)
                    accelerator.log({"image": image}, step=global_step)

        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        logger.info("Saving final model")
        save_model(config, text_encoder, unet, global_step, accelerator)

    accelerator.end_training()


def load_tokenizer(config: TrainConfig) -> CLIPTokenizer:
    return CLIPTokenizer.from_pretrained(config.base_model_name_or_path, subfolder="tokenizer")


def load_text_encoder(config: TrainConfig) -> CLIPTextModel:
    text_encoder = CLIPTextModel.from_pretrained(
        config.base_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float32 if config.train_text_encoder else config.torch_dtype,
        device_map="auto",
    )
    if not config.train_text_encoder:
        text_encoder.requires_grad_(False)
        text_encoder.eval()
    if config.train_text_encoder and config.gradient_checkpointing:
        text_encoder.gradient_checkpointing_enable()
    return text_encoder


def load_vae(config: TrainConfig) -> AutoencoderKL:
    vae = AutoencoderKL.from_pretrained(
        config.base_model_name_or_path, subfolder="vae", torch_dtype=config.torch_dtype, device_map="auto"
    )
    vae.enable_xformers_memory_efficient_attention()
    vae.requires_grad_(False)
    vae.eval()
    return vae


def load_unet(config: TrainConfig) -> UNet2DConditionModel:
    unet = UNet2DConditionModel.from_pretrained(
        config.base_model_name_or_path,
        subfolder="unet",
        torch_dtype=torch.float32 if config.train_unet else config.torch_dtype,
        device="auto",
    )
    unet.enable_xformers_memory_efficient_attention()
    if not config.train_unet:
        unet.requires_grad_(False)
        unet.eval()
    if config.train_unet and config.gradient_checkpointing:
        unet.enable_gradient_checkpointing()
    return unet


def load_optimizer(config: TrainConfig, parameters):
    if config.use_8bit_adam:
        return bnb.optim.AdamW8bit(parameters, lr=config.learning_rate)
    else:
        return torch.optim.AdamW(parameters, lr=config.learning_rate)


def save_model(
    config: TrainConfig, text_encoder: CLIPTextModel, unet: UNet2DConditionModel, step: int, accelerator: Accelerator
):
    """Save the models for inference.

    Only trained models are saved. The base checkpoints will be re-loaded during inference for frozen models.
    """
    checkpoint_path = config.checkpoint_dir / f"step_{step}"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if config.train_unet:
        accelerator.unwrap_model(unet).save_pretrained(checkpoint_path / "unet")
    if config.train_text_encoder:
        accelerator.unwrap_model(text_encoder).save_pretrained(checkpoint_path / "text_encoder")


def generate_samples(
    vae: AutoencoderKL,
    text_encoder: CLIPTextModel,
    tokenizer: CLIPTokenizer,
    unet: UNet2DConditionModel,
    noise_scheduler: DDPMScheduler,
    config: TrainConfig,
    accelerator: Accelerator,
) -> wandb.Image:
    """Generate samples from the fine-tuned models, tiled as a single image."""
    pipeline = StableDiffusionPipeline.from_pretrained(
        config.base_model_name_or_path,
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        unet=unet,
        scheduler=noise_scheduler,
        safety_checker=None,
        requires_safety_checker=False,
        device_map="auto",
        torch_dtype=config.torch_dtype,
    )
    pipeline.set_progress_bar_config(disable=True)

    images = []
    for _ in range(config.eval_n_generate_samples):  # using batch size 1 to prevent OOM on T4
        with accelerator.autocast():
            output: StableDiffusionPipelineOutput = pipeline(
                prompt=config.target_prompt, num_inference_steps=config.eval_num_inference_steps, output_type="np"
            )
        images.append(output.images[0])

    image = tile_images(images)
    image = pipeline.numpy_to_pil(image)[0]

    return wandb.Image(image, caption=config.target_prompt)


if __name__ == "__main__":
    main()
