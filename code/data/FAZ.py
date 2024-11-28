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
from PIL import Image
from torch.utils.data import Dataset
from data.common_utils import get_image_pairs


class FAZDataset(Dataset):
    def __init__(self, root_dir, mode, transform=None):
        self.img_mask_pairs = get_image_pairs(root_dir, mode)
        self.transform = transform

    def __len__(self):
        return len(self.img_mask_pairs)

    def __getitem__(self, idx):
        img_path, mask_path = self.img_mask_pairs[idx]
        image = Image.open(img_path).convert('L')
        
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
