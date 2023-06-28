from typing import Literal, Union

import pytorch_lightning as pl
import torch
from torch import nn


class DigitRecognizer(pl.LightningModule):
    def __init__(self, loss_fn: Union[Literal["cross_entropy", "mse"]], learning_rate: float, num_classes: int = 10):
        super().__init__()
        self.save_hyperparameters()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, 3, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 256, 3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 1),
            nn.Flatten(-3, -1),
            nn.Linear(256 * 5 * 5, num_classes),
        )
        losses = {"cross_entropy": nn.CrossEntropyLoss(), "mse": nn.MSELoss()}
        self.loss_fn = losses[loss_fn]
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        samples, labels = batch
        predictions = self(samples).reshape(-1, self.num_classes)
        loss = self.loss_fn(predictions, labels)
        return loss

    def test_step(self, batch, batch_idx):
        samples, labels = batch
        predictions = self(samples).reshape(-1, self.num_classes)
        loss = self.loss_fn(predictions, labels)
        self.log("test_loss", loss, on_step=True, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=self.learning_rate)
        return optimizer
