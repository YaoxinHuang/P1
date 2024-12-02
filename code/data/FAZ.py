#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:FAZ.py
# author:xm
# datetime:2024/10/26 20:29
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import random, torch, glob
from PIL import Image
from torch.utils.data import Dataset
from data.common_utils import get_image_pairs
from data.fft import *
from scipy.ndimage import distance_transform_edt
from yaoxin_tools import mapping


class FAZDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None, domain_amp_dir=None, shift_ratio=0.3):
        self.img_mask_pairs, self.labels = get_image_pairs(root_dir, mode)
        self.transform = transform
        self.mode = mode
        self.domain_amp_dir = r'../domain_amp' if domain_amp_dir is None else domain_amp_dir
        # self.domain1 = np.load(self.domain_amp_dir + '/domain1.npy')
        self.domain_amps = []
        for i in range(1,6):
            self.domain_amps.extend(glob.glob(f"../FAZ/Domain{i}/train/imgs/*.png"))
            # self.domain_amps.extend(glob.glob(f"D:/P1/FAZ/Domain{i}/train/imgs/*.png"))
        self.shift_ratio = shift_ratio

    def _get_np_img(self, img_dir):
        img = np.asarray(Image.open(img_dir).convert('L'))
        return img
    
    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, idx):
        if self.mode == 'train':
            img_path, mask_path = self.img_mask_pairs[idx]
            img = self._get_np_img(img_path)
            mask = np.where(self._get_np_img(mask_path)>0, 255, 0)
            if np.random.rand() < self.shift_ratio:
                shift_img = np.clip(domain_shift(img, self._get_np_img(random.choice(self.domain_amps))), 0, 255)
            else:
                shift_img = img
            if self.transform:
                shift_img = self.transform(Image.fromarray(shift_img.astype(np.uint8)))
                mask = self.transform(Image.fromarray(mask.astype(np.uint8)))
            return shift_img, mask
        else:
            img_path, mask_path = self.img_mask_pairs[idx]
            image = Image.open(img_path).convert('L')
            mask = Image.open(mask_path).convert('L')
            if self.transform:
                image = self.transform(image)
                mask = self.transform(mask)
            return image, mask