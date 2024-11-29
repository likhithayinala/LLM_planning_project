import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from detection_model import select_det_model
from main_model import select_main_model
from dataloader import dataset
import logging
import wandb
import os
import json
import argparse

def set_config(args):
    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    config.update({
        "detection_model": args.detection_model,
        "main_model": args.main_model,
        "dataset": args.dataset,
        "wandb": args.wandb,
        "log": args.log,
        "tag": args.tag,
        "debug": args.debug
    })
    if config['debug']:
        config.update({'wandb': False, 'log': False})
    # Setup results directory
    result_path = os.path.join(config['result_path'], f"{config['detection_model']}_{config['main_model']}_{config['dataset']}_{config['tag']}")
    os.makedirs(result_path, exist_ok=True)
    logging.basicConfig(filename=os.path.join(result_path, 'results.log'), level=logging.INFO)
    with open(os.path.join(result_path, 'config.json'), 'w') as f:
        json.dump(config, f)

    # Setup W&B if enabled
    if config['wandb']:
        wandb.init(project='detection', config=config)
    return config

class DetectionTrainingModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config

        # Initialize models
        self.detection_model = select_det_model(config['detection_model'], config)
        self.main_model = select_main_model(config['main_model'], config)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        state = self.main_model(x)
        return self.detection_model(state)

    def training_step(self, batch, batch_idx):
        data, labels = batch
        outputs = self(data)
        loss = self.criterion(outputs, labels)

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == labels).sum().item() / labels.size(0)

        # Log training metrics
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_accuracy", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def configure_optimizers(self):
        # Optimizer
        optimizer = torch.optim.Adam(self.detection_model.parameters(), lr=self.config['lr'])
        return optimizer


class DataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        self.train_dataset = dataset(self.config)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, default='config.json')
    parser.add_argument('--detection_model', '-dm', type=str, default='MLP')
    parser.add_argument('--main_model', '-mm', type=str, default='Llama2')
    parser.add_argument('--dataset', '-ds', type=str, default='CIFAR10')
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--wandb', type=bool, default=True)
    parser.add_argument('--log', type=bool, default=True)
    parser.add_argument('--tag', '-t', type=str, default='')
    args = parser.parse_args()

    config = set_config(args)

    # Initialize Lightning modules
    model = DetectionTrainingModule(config)
    data_module = DataModule(config)

    # Train
    trainer = pl.Trainer(
        max_epochs=config['epochs'],
        log_every_n_steps=1,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=config['result_path'],
                filename='best_model',
                save_top_k=1,
                monitor="train_accuracy",
                mode="max"
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='step'),
        ],
    )
    trainer.fit(model, data_module)

    # End W&B session
    if config['wandb']:
        wandb.finish()
