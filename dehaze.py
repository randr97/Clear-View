import argparse
import glob
import os
import sys
import time

import net
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision
from PIL import Image
from torchvision import transforms

import dataloader


def dehaze_image(image_path):

    data_hazy = Image.open(image_path)
    data_hazy = (np.asarray(data_hazy) / 255.0)

    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1)
    data_hazy = data_hazy.cuda().unsqueeze(0)

    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('snapshots/dehazer.pth'))

    clean_image = dehaze_net(data_hazy)
    torchvision.utils.save_image(
        torch.cat((data_hazy, clean_image), 0), "results/" + image_path.split("/")[-1])


if __name__ == '__main__':

    test_list = glob.glob("test_images/*")

    for image in test_list:

        dehaze_image(image)
        print(image, "done!")
