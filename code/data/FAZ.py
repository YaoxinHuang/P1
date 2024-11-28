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
import cv2
from torch.utils.data import Dataset
from data.common_utils import get_image_pairs
from data.fft import *


class FAZDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None, domain_amp_dir=None, shift_ratio=0.6):
        self.img_mask_pairs, self.labels = get_image_pairs(root_dir, mode)
        self.transform = transform
        self.domain_amp_dir = r'../domain_amp' if domain_amp_dir is None else domain_amp_dir
        self.domain1 = np.load(self.domain_amp_dir + '/domain1.npy')
        self.domain2 = np.load(self.domain_amp_dir + '/domain2.npy')
        self.domain3 = np.load(self.domain_amp_dir + '/domain3.npy')
        self.domain4 = np.load(self.domain_amp_dir + '/domain4.npy')
        self.domain5 = np.load(self.domain_amp_dir + '/domain5.npy')
        self.domain_amps = [self.domain1, self.domain2, self.domain3, self.domain4, self.domain5]
        self.shift_ratio = shift_ratio

    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.img_mask_pairs[idx]
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE).permute(2, 0, 1)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE).permute(2, 0, 1)
        if np.random.rand() < self.shift_ratio:
            shift_img = domain_shift(image, np.random.choice(self.domain_amps))
            return shift_img, mask
        else:
            return image, mask
