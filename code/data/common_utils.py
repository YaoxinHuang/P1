#!/usr/bin/env python
# -*- coding:utf-8 -*-

# file:common_utils.py
# author:xm
# datetime:2024/10/26 20:29
# software: PyCharm

"""
this is function  description 
"""

# import module your need
import os


def get_image_pairs(root_dir, mode):
    img_mask_pairs = []
    for domain in os.listdir(root_dir):
        if mode == 'train' and domain != 'Domain1':
            continue
        domain_path = os.path.join(root_dir, domain)
        split_path = os.path.join(domain_path, mode)
        imgs_path = os.path.join(split_path, 'imgs')
        masks_path = os.path.join(split_path, 'mask')
        for img_name in os.listdir(imgs_path):
            if img_name.endswith('.png'):
                img_path = os.path.join(imgs_path, img_name)
                mask_path = os.path.join(masks_path, img_name)
                img_mask_pairs.append((img_path, mask_path))
    return img_mask_pairs
