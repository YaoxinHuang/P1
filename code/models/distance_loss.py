from torch import nn as nn
from scipy.ndimage import distance_transform_edt
import torch
import numpy as np


class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # Apply sigmoid to inputs if they are logits
        inputs = torch.sigmoid(inputs)
        
        # Flatten the tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        union = inputs.sum() + targets.sum()
        
        # Calculate Dice coefficient
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        
        # Return Dice loss
        return 1 - dice

class Distance_Loss(nn.Module):
    def __init__(self):
        super(Distance_Loss, self).__init__()
        self.alpha = .6
        self.beta = 1
    
    def forward(self, inputs, targets):
        weight = torch.zeros_like(targets)
        device = targets.device
        eps = 1e-3
        for b in range(targets.shape[0]):
            # in numpy
            w = distance_transform_edt(targets[b, ...].squeeze().cpu().numpy())
            v = np.max(w)
            mask = (w != 0)[np.newaxis, ...]
            w = v - w[b, ...]
            w = w * mask

            # in torch
            weight[b, ...] = torch.tensor(w).to(device)
            weight[b, ...] += eps
        
        # MSE Loss
        self.weightMSELoss = torch.mean(weight*(inputs-targets)**2)

        #Dice Loss
        self.DiceLoss = DiceLoss()

        # return self.alpha * self.weightMSELoss + self.beta * self.DiceLoss(inputs, targets)
        return self.alpha * self.weightMSELoss, self.beta * self.DiceLoss(inputs, targets)

class Distance_Loss_copy(nn.Module):
    def __init__(self):
        super(Distance_Loss_copy, self).__init__()
        self.alpha = 1000
        self.beta = 1
    
    def forward(self, inputs, targets):
        weight = torch.zeros_like(targets)
        device = targets.device
        eps = 1e-3
        for b in range(targets.shape[0]):
            # in numpy
            w = distance_transform_edt(targets[b, ...].squeeze().cpu().numpy())
            v = np.max(w)
            mask = (w != 0)[np.newaxis, ...]
            w = v - w[b, ...]
            w = w * mask

            # in torch``
            weight[b, ...] = torch.tensor(w).to(device)
            weight[b, ...] = weight[b, ...] / ((torch.sum(weight[b, ...])) + eps) * 10
        
        # MSE Loss
        self.weightMSELoss = torch.mean(weight*(inputs-targets)**2)

        #Dice Loss
        self.DiceLoss = DiceLoss()

        # return self.alpha * self.weightMSELoss
        return self.alpha * self.weightMSELoss, self.beta * self.DiceLoss(inputs, targets)
