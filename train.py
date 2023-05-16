import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Train:
    def __init__(self, net, trainset, valset, config):
        self.net = net.to(device=device)
        self.net.apply(weights_init)
        self.trainset = trainset
        self.valset = valset
        self.config = config
        self.criterion = nn.MSELoss().to(device=device)
        self.optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay
        )
        self.net.train()
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device=device)
        self.psnr = PeakSignalNoiseRatio()
        self.ssim_values = []
        self.psnr_values = []
        self.training_times = []
        self.iteration_losses = []
        self.validation_losses = []

    def train(self):
        print("Initiating Training...")
        print("Device: ", device)
        print(f"Trainable Parameters: {sum(p.numel() for p in self.net.parameters() if p.requires_grad)}")
        for epoch in range(self.config.num_epochs):
            train_loss = 0.00
            valid_loss = 0.00
            train_start = time.monotonic()
            print(f"Running Training Stage for epoch: {epoch}")
            for iteration, (img_orig, img_haze) in enumerate(self.trainset):
                img_orig = img_orig.to(device=device)
                img_haze = img_haze.to(device=device)
                clean_image = self.net(img_haze)
                loss = self.criterion(clean_image, img_orig)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                self.iteration_losses.append(loss.item())
                train_loss += loss.item()

                # Checkpoint is every 50 iterations.
                if ((iteration + 1) % 50) == 0:
                    print("Train Loss at iteration", iteration + 1, ":", loss.item())
            train_end = time.monotonic()
            print("Running Validation Stage...")
            self.net.eval()
            with torch.no_grad():
                for iter_val, (img_orig, img_haze) in enumerate(self.valset):
                    img_orig = img_orig.to(device=device)
                    img_haze = img_haze.to(device=device)
                    clean_image = self.net(img_haze)
                    val_loss = self.criterion(clean_image, img_orig)
                    self.validation_losses.append(val_loss.item())
                    valid_loss += val_loss

                    # Checkpoint is every 10 steps.
                    if ((iter_val + 1) % 10) == 0:
                        print("Validation Loss at iteration", iter_val + 1, ":", val_loss.item())

            # Sample image to calculate PSNR and SSIM scores.
            img_orig, img_haze = next(iter(self.valset))
            img_orig = img_orig.to(device=device)
            img_haze = img_haze.to(device=device)
            clean_image = self.net(img_haze)

            # Interpolation for torchmetrics classes.
            cpu_clean = clean_image[0].cpu().unsqueeze(0)
            cpu_original = img_orig[0].cpu().unsqueeze(0)

            # PSNR and SSIM
            psnr_epoch = self.psnr(np.squeeze(cpu_clean), np.squeeze(cpu_original))
            ssim_epoch = self.ssim(clean_image, img_orig)

            # Logging.
            self.ssim_values.append(ssim_epoch.item())
            self.psnr_values.append(psnr_epoch.item())
            self.training_times.append(train_end - train_start)
            avg_train_loss = train_loss / len(self.trainset)
            avg_val_loss = valid_loss / len(self.valset)
            self.save_epoch_data()
            print(
                f"Epoch: {epoch}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}\nSSIM: {ssim_epoch},"
                f"PSNR: {psnr_epoch}, Training Time: {train_end - train_start}"
            )

    def save_epoch_data(self):
        torch.save(self.net.state_dict(), self.config.snapshots_folder + self.config.run_id)
        with open(self.config.run_id, 'w') as f:
            json.dump({
                'ssim_values': self.ssim_values,
                'psnr_values': self.psnr_values,
                'training_times': self.training_times,
                'iteration_losses': self.iteration_losses,
                'validation_losses': self.validation_losses,
            }, f)
