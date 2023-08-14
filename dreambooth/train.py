import os
import random
from config import DreamboothConfigTrain
import shutil


def rename_instance_images(config_train: DreamboothConfigTrain):
    """Renames the instance images in the training dataset in the format <instance_name> (<number>).<extension>."""
    directory = config_train.instance_data_dir

    # only picks files with the following extensions 
    allowed_extensions = {".png", ".jpg", ".jpeg"}
    files = [img_file for img_file in sorted(os.listdir(directory)) if os.path.splitext(img_file)[-1] in allowed_extensions]
    files.sort()

    for count, filename in enumerate(files):
        _, extension = os.path.splitext(filename)

        new_filename = f"{config_train.instance_name} ({count}){extension}"

        current_path = os.path.join(directory, filename)
        new_path = os.path.join(directory, new_filename)

        os.rename(current_path, new_path)

    print("Images renamed successfully")


def setup_configs_train(config_train: DreamboothConfigTrain):
    """Sets up the training configurations."""
    if config_train.seed == '':
        config_train.seed = random.randint(1, 999999)
    else:
        config_train.seed = int(config_train.seed)

    print("Seed: ", config_train.seed)
    print("Reuse this seed if needed")

    config_train.model_saved_output_dir

    unet_model_dir = os.path.join(
        config_train.model_saved_output_dir, 'unet/diffusion_pytorch_model.bin'
    )

    if config_train.resume_training:
        if os.path.exists(unet_model_dir):
            config_train.pretrained_model_name_or_path = (
                config_train.model_saved_output_dir
            )
            print('Resuming Training...')
        else:
            print('Previous model not found, training a new model...')
            config_train.pretrained_model_name_or_path = config_train.base_model_path

    print(f"{config_train.text_encoder_training_steps=}")

    if config_train.text_encoder_training_steps == 0:
        config_train.enable_text_encoder_training = False


def train(config_train: DreamboothConfigTrain):
    """Trains the Text Encoder model and the UNet.
    It has the capability to resume training from a checkpoint. The training can be resumed if the config variable RESUME_TRAINING is set to True.
    """

    if config_train.enable_text_encoder_training:
        print('Training the text encoder...')
        text_encoder_trained_path = os.path.join(
            config_train.output_dir, 'text_encoder_trained'
        )
        if os.path.exists(text_encoder_trained_path):
            shutil.rmtree(text_encoder_trained_path)
        train_text_encoder(config_train)

    if config_train.unet_training_steps != 0:
        print('Training the UNet...')
        train_unet(config_train)


def train_text_encoder(config_train: DreamboothConfigTrain):
    """Trains the text encoder from the model."""
    # TODO: Avoid the use of subshell to call train_dreambooth.py and call it directly through the Run command.
    
    # Flag is set such that if training has been resumed, VAE should be loaded from the output model directory else it
    # should be loaded from 'stabilityai/sd-vae-ft-mse'
    if config_train.resume_training:
        resume_training = "--resume_training"
    else:
        resume_training = ""

    command = f"accelerate launch train_dreambooth.py \
                --image_captions_filename \
                 --train_text_encoder \
                 --dump_only_text_encoder \
                 --pretrained_model_name_or_path={config_train.pretrained_model_name_or_path} \
                 --pretrained_vae_path={config_train.pretrained_vae_path} \
                 --instance_data_dir={config_train.instance_data_dir} \
                 --output_dir={config_train.output_dir} \
                 --captions_dir={config_train.captions_dir} \
                 --seed={config_train.seed} \
                 --resolution={config_train.resolution} \
                 --mixed_precision={config_train.mixed_precision} \
                 --train_batch_size=1 \
                 --gradient_accumulation_steps=1 --gradient_checkpointing \
                 --use_8bit_adam \
                 --learning_rate={config_train.text_encoder_learning_rate} \
                 --lr_scheduler='linear' \
                 --lr_warmup_steps=0 \
                 {resume_training} \
                 --max_train_steps={config_train.text_encoder_training_steps}"
    os.system(command)


def train_unet(config_train: DreamboothConfigTrain):
    """Trains the UNet from the model. It can resume training if needed."""
    # TODO: Avoid the use of subshell to call train_dreambooth.py and call it directly through the Run command.

    # Flag is set such that if training has been resumed, VAE should be loaded from the output model directory else it
    # should be loaded from 'stabilityai/sd-vae-ft-mse'
    if config_train.resume_training:
        resume_training = "--resume_training"
    else:
        resume_training = ""

    command = f"accelerate launch train_dreambooth.py \
                 --image_captions_filename \
                 --train_only_unet \
                 --save_starting_step={config_train.save_starting_step} \
                 --save_n_steps={config_train.save_n_steps} \
                 --Session_dir={config_train.output_dir} \
                 --pretrained_model_name_or_path={config_train.pretrained_model_name_or_path} \
                 --pretrained_vae_path={config_train.pretrained_vae_path} \
                 --instance_data_dir={config_train.instance_data_dir} \
                 --output_dir={config_train.output_dir} \
                 --captions_dir={config_train.captions_dir} \
                 --seed={config_train.seed} \
                 --resolution={config_train.resolution} \
                 --mixed_precision={config_train.mixed_precision} \
                 --train_batch_size=1 \
                 --gradient_accumulation_steps=1 \
                 --train_batch_size=1 \
                 --gradient_accumulation_steps=1 \
                 --use_8bit_adam \
                 --learning_rate={config_train.unet_learning_rate} \
                 --lr_scheduler='linear' \
                 --lr_warmup_steps=0 \
                 {resume_training} \
                 --max_train_steps={config_train.unet_training_steps}"
    os.system(command)


def main(config_train: DreamboothConfigTrain):
    rename_instance_images(config_train)
    setup_configs_train(config_train)
    train(config_train)


if __name__ == "__main__":
    config_train: DreamboothConfigTrain = DreamboothConfigTrain.parse_raw(
        os.environ["CONFIG"]
    )

    main(config_train)
