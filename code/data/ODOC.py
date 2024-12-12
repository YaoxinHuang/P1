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
import random, torch, glob, sys
sys.path.insert(0, 'D:/Course/P1/code')
from PIL import Image
from torch.utils.data import Dataset

from data.common_utils import get_image_pairs
from data.fft import *
from scipy.ndimage import distance_transform_edt
from yaoxin_tools import mapping


class ODOCDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None, shift_ratio=0.7, alpha=0.05):
        self.img_mask_pairs, self.labels = get_image_pairs(root_dir, mode)
        self.transform = transform
        self.mode = mode
        self.domain_amps = []
        for i in range(1,6):
            self.domain_amps.extend(glob.glob(f"../ODOC/Domain{i}/train/imgs/*.png"))
        self.shift_ratio = shift_ratio
        self.alpha = alpha

    def _get_np_img(self, img_dir, mode='RGB'):
        img = np.array(Image.open(img_dir).convert(mode))
        return img
    
    def __len__(self):
        return len(self.img_mask_pairs)
    
    def _get_cls_mask(self, mask, num_cls):
        mask1 = np.zeros((mask.shape[0], mask.shape[1], num_cls))
        cls = np.unique(mask)
        for i in range(num_cls):
            mask1[..., i] = np.where(mask==cls[i], 1, 0)
        return mask1

    def __getitem__(self, idx):
        img_path, mask_path = self.img_mask_pairs[idx]
        mask = self._get_np_img(mask_path, mode='L')
        mask = self._get_cls_mask(mask, 3).transpose(2, 0, 1)
        # mask is ont-hot category label in 3 dimension

        if self.mode == 'train':
            img = self._get_np_img(img_path)
            if np.random.rand() < self.shift_ratio:
                shift_img = np.clip(domain_shift(img, self._get_np_img(random.choice(self.domain_amps)), axes=(0, 1), alpha=self.alpha), 0, 255)
            else:
                shift_img = img
            if self.transform:
                shift_img = self.transform(Image.fromarray(shift_img.astype(np.uint8)))
            return shift_img, mask
        else:
            img_path, mask_path = self.img_mask_pairs[idx]
            image = self._get_np_img(img_path)
            if self.transform:
                image = self.transform(image)
            return image, mask 

if __name__ == '__main__':
    data = ODOCDataset(r'../../ODOC', 'train', 0.1)
    print(len(data))
    img, mask = data.__getitem__(4)
    print(np.unique(mask))