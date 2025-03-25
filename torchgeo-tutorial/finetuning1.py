import os
import tempfile

import timm
import torch
from lightning.pytorch import Trainer

from torchgeo.datamodules import EuroSAT100DataModule
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask

# variables for training
batch_size = 10
max_epochs = 3
fast_dev_run = False


# Torchgeo's lightining datamodule

root = os.path.join(tempfile.gettempdir(), 'eurosat100')
datamodule = EuroSAT100DataModule(
    root=root, batch_size=batch_size, download=True
)

# Using torchgeo's WeightEnum
weights = ResNet18_Weights.SENTINEL2_ALL_MOCO

# Torchgeo's trainer for a specific downstream task, classification in this case

task = ClassificationTask(
    model='resnet18',
    loss='ce',
    weights=weights,
    in_channels=13,
    num_classes=10,
    lr=0.001,
    patience=5,
)


# Train using pytorch lighning's Trainer

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
default_root_dir = os.path.join(tempfile.gettempdir(), 'experiments')

trainer = Trainer(
    accelerator=accelerator,
    default_root_dir=default_root_dir,
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    min_epochs=1,
    max_epochs=max_epochs,
)

trainer.fit(model=task, datamodule=datamodule)