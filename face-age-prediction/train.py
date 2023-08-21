import os
from pathlib import Path
from typing import Tuple

import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision
import wandb
from torch import nn
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from dataset import FaceDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train(train_loader: DataLoader, model: nn.Module, criterion: nn.Module, optimizer: nn.Module) -> tuple[
    float, float]:
    """Perform one epoch's training"""
    model.train()
    total_loss = 0.0
    train_acc = 0.0
    num_examples = 0

    # train_loader using tqdm
    for x, y in tqdm(train_loader):
        x, y = x.to(DEVICE), y.float().to(DEVICE)

        # compute output
        outputs = model(x)
        outputs = outputs.softmax(-1)

        ages = torch.arange(0, 101).to('cuda')
        outputs = (outputs * ages).sum(axis=-1)

        # compute loss
        loss = criterion(outputs, y)
        wandb.log({"train_loss": loss.item()})

        # measure accuracy and record loss
        with torch.no_grad():
            train_acc += torch.sum(torch.round(outputs) == y).item()
        total_loss += loss.item() * len(x)
        num_examples += len(x)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # clip gradient values to avoid exploding gradient problem
        nn.utils.clip_grad_value_(model.parameters(), 1.0)

        optimizer.step()

    total_loss /= num_examples

    return total_loss, train_acc / (num_examples)


@torch.inference_mode()
def validate(validate_loader: DataLoader, model: nn.Module) -> float:
    """Perform validation with the current epoch's model on the validation set"""
    model.eval()

    total_loss = 0.0
    num_samples = 0

    for x, y in tqdm(validate_loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        # compute output
        outputs = model(x)
        outputs = outputs.softmax(-1)
        ages = torch.arange(0, 101).to('cuda').float()
        outputs = (outputs * ages).sum(axis=-1)  # Calculate the expected value of the distribution of ages

        loss = F.l1_loss(outputs, y, reduction='sum')  # Calculate the mean absolute error

        total_loss += loss.item()
        num_samples += x.size(0)

    mae = total_loss / num_samples

    return mae


def main(config_train: Config):
    if DEVICE == "cuda":
        cudnn.benchmark = True  # TODO: What's this?

    start_epoch = 0  # ??

    # Get Model ready
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, 101)
    model = model.to(DEVICE)

    # Setting up Wandb
    wandb.init(
        config={
            "optimizer": config_train.optimizer,
            "learning rate": config_train.learning_rate,
            "momentum": config_train.momentum,
            "weight decay": config_train.weight_decay,
        },
    )

    # Setting up Optimizer
    if config_train.optimizer == "sgd":
        print("Using SGD optimizer")
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config_train.learning_rate,
            momentum=config_train.momentum,
            weight_decay=config_train.weight_decay,
        )
    else:
        print("Using Adam optimizer")
        optimizer = torch.optim.Adam(
            model.parameters(), lr=config_train.learning_rate, weight_decay=config_train.weight_decay
        )

    # Setting up model weights to restart training from a given checkpoint
    if config_train.resume_path:
        if Path(config_train.resume_path).is_file():
            checkpoint = torch.load(config_train.resume_path, map_location=DEVICE)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Setting up the learning rate scheduler
    scheduler = StepLR(
        optimizer,
        step_size=config_train.learning_rate_decay_step,
        gamma=config_train.learning_rate_decay_rate,
        last_epoch=start_epoch - 1,
    )

    criterion = nn.MSELoss().to(DEVICE)

    # Used to log the weights and gradients of the model parameters
    wandb.watch(model, criterion, log="all", log_freq=50)

    # Setting up the training dataset and dataloader
    train_dataset = FaceDataset(
        config_train.data_dir, "train", img_size=config_train.img_size, augment=True, age_stddev=config_train.age_stddev
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config_train.batch_size,
        shuffle=True,
        num_workers=config_train.num_workers,
        drop_last=True,
    )

    # Setting up the validation dataset and dataloader
    val_dataset = FaceDataset(config_train.data_dir, "valid", img_size=config_train.img_size, augment=False)
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_train.batch_size,
        shuffle=False,
        num_workers=config_train.num_workers,
        drop_last=False,
    )

    best_val_mae = float("inf")

    for epoch in range(start_epoch, config_train.epochs):
        train_loss, train_accuracy = train(train_loader, model, criterion, optimizer)

        val_mae = validate(val_loader, model)

        print(f"Epoch: {epoch} | train loss: {train_loss} | train_accuracy: {train_accuracy} | val mae: {val_mae}")
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "val_mae": val_mae})

        # update learning rate
        scheduler.step()
        wandb.log({"lr": scheduler.get_last_lr()[0]})

        # Save the model with the best validation MAE
        if val_mae < best_val_mae:
            checkpoint_name = f"best_mae.pth"
            checkpoint_dir_path = os.path.join(config_train.checkpoint, checkpoint_name)
            print(f"Saving best checkpoint to {checkpoint_dir_path}")

            torch.save(  # TODO: Save to separate files
                {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                checkpoint_dir_path,
            )
            best_val_mae = val_mae

    wandb.finish()

    print("=> training finished")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    config_train = Config.model_validate_json(os.environ.get("CONFIG", "{}"))

    main(config_train)
