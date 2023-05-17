import argparse
import os

import torch

import dataloader
import model
from train import Train


class Driver:

    def __init__(self, config):
        self.config = config

    def run(self):
        print(f'Running model: {config.model}')
        if config.model == "clearview":
            net = model.ClearView()
        elif config.model == "dehaze":
            net = model.LightDehazeNet()
        else:
            raise Exception('Model does not exist')
        trainset = torch.utils.data.DataLoader(
            dataloader.DehazingLoader(
                config.og_img,
                config.hazy_img
            ),
            batch_size=config.train_batch,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True
        )
        valset = torch.utils.data.DataLoader(
            dataloader.DehazingLoader(
                config.og_img,
                config.hazy_img,
                train=False
            ),
            batch_size=config.val_batch,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True
        )
        Train(net, trainset, valset, config).train()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    torch.backends.cudnn.benchmark = True
    # Input Parameters
    parser.add_argument('--og_img', type=str, default="data/original/")
    parser.add_argument('--hazy_img', type=str, default="data/haze/")
    parser.add_argument('--lr', type=float, default=0.0004)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--grad_clip_norm', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--train_batch', type=int, default=16)
    parser.add_argument('--val_batch', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
    parser.add_argument('--model', type=str, default="clearview")
    parser.add_argument('--run_id', type=str, default="clearview")
    parser.add_argument('--logs', type=str, default="logs/")

    config = parser.parse_args()

    if not os.path.exists(config.snapshots_folder):
        os.mkdir(config.snapshots_folder)
    if not os.path.exists(config.logs):
        os.mkdir(config.logs)

    driver = Driver(config)
    driver.run()
