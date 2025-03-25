import os
import tempfile

import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from torchgeo.datamodules import EuroSAT100DataModule
from torchgeo.models import ResNet18_Weights
from torchgeo.trainers import ClassificationTask

# similary as in inference.py

batch_size = 10
max_epochs = 50
fast_dev_run = False

root = os.path.join(tempfile.gettempdir(), 'eurosat100')
datamodule = EuroSAT100DataModule(
    root=root, batch_size=batch_size, download=True
)


task = ClassificationTask(
    loss='ce',
    model='resnet18',
    weights=ResNet18_Weights.SENTINEL2_ALL_MOCO,
    in_channels=13,
    num_classes=10,
    lr=0.1,
    patience=5,
)



# Training sanity setup

accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
default_root_dir = os.path.join('tensorboard_logs', 'experiments')
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss', dirpath=default_root_dir, save_top_k=1, save_last=True
)
early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0.00, patience=10)
logger = TensorBoardLogger(save_dir=default_root_dir, name='tutorial_logs')


# Pytorch lightining's way of training model

trainer = Trainer(
    accelerator=accelerator,
    callbacks=[checkpoint_callback, early_stopping_callback],
    fast_dev_run=fast_dev_run,
    log_every_n_steps=1,
    logger=logger,
    min_epochs=1,
    max_epochs=max_epochs,
)

trainer.fit(model=task, datamodule=datamodule)

# run below command on terminal to view train logs
# tensorboard --logdir "tensorboard_logs" 

trainer.test(model=task, datamodule=datamodule)
