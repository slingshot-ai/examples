import torch
import pytorch_lightning as pl
from torch.nn.functional import mse_loss, l1_loss
from torch import nn, optim
import torchvision
from torchvision.models import ResNet18_Weights


class FaceAgeTrainer(pl.LightningModule):
    def __init__(self, learning_rate: float | None = None):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.model.fc = nn.Linear(self.model.fc.in_features, 101)

        # Training Hyper-parameters are only supposed to be set to None when running inference
        self.learning_rate = learning_rate

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        face_images, ages = batch
        expected_ages = self.forward(face_images)
        loss = mse_loss(expected_ages, ages)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        face_images, ages = batch
        expected_ages = self.forward(face_images)
        mae_loss_value = l1_loss(expected_ages, ages)
        mse_loss_value = mse_loss(expected_ages, ages)
        self.log("val/MAE_loss", mae_loss_value)
        self.log("val/MSE_loss", mse_loss_value)
        return mse_loss_value

    def configure_optimizers(self) -> torch.optim.Optimizer:
        assert self.learning_rate, "Learning rate must be set when training"
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def forward(self, face_images: torch.Tensor) -> torch.Tensor:
        outputs = self.model(face_images)
        outputs = outputs.softmax(-1)  # Shape is (batch_size, 101)

        # Let's compute the expected value of the distribution of ages
        # We'll use that as our predicted age to compute the loss
        age_range = torch.arange(0, 101).to(self.device)
        expected_ages = (outputs * age_range).sum(dim=-1)
        return expected_ages
