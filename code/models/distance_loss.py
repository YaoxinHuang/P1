from torch import nn as nn

import torch


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
        self.beta = .4

        self.MSELoss = nn.MSELoss()
        self.DiceLoss = DiceLoss()
    
    def forward(self, inputs, targets):
        return self.alpha * self.MSELoss(inputs, targets) + self.beta * self.DiceLoss(inputs, targets)