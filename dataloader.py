import glob
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image


class DehazingLoader(data.Dataset):

    def __init__(self, og_img, hazy_img, train=True):
        self.train_list, self.val_list = self.populate_data(og_img, hazy_img)
        if train:
            self.data_list = self.train_list
            print("Total training examples:", len(self.train_list))
        else:
            self.data_list = self.val_list
            print("Total validation examples:", len(self.val_list))

    def __getitem__(self, index):
        data_orig_path, data_hazy_path = self.data_list[index]

        data_orig = Image.open(data_orig_path)
        data_hazy = Image.open(data_hazy_path)

        data_orig = data_orig.resize((480, 640), Image.ANTIALIAS)
        data_hazy = data_hazy.resize((480, 640), Image.ANTIALIAS)

        data_orig = (np.asarray(data_orig) / 255.0)
        data_hazy = (np.asarray(data_hazy) / 255.0)

        data_orig = torch.from_numpy(data_orig).float()
        data_hazy = torch.from_numpy(data_hazy).float()
        return data_orig.permute(2, 0, 1), data_hazy.permute(2, 0, 1)

    def __len__(self):
        return len(self.data_list)

    def populate_data(self, og_img, hazy_img, train_test_split=0.9):
        haze_list = glob.glob(hazy_img + "*.jpg")
        filestruct = defaultdict(list)
        for i in haze_list:
            img = i.split("/")[-1]
            k = img.split('_')
            filestruct[f"{k[0]}_{k[1]}.jpg"].append(img)

        ct, train_list, val_list = 0, [], []
        for k, v in filestruct.items():
            for each in v:
                if ct < len(filestruct.keys()) * train_test_split:
                    train_list.append([og_img + k, hazy_img + each])
                else:
                    val_list.append([og_img + k, hazy_img + each])
            ct += 1
        return train_list, val_list
