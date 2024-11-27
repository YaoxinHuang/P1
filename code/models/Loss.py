import torch
import torch_topological.nn as ttnn

from torch import nn as nn
from yaoxin_tools import timeit, usual_reader


class TopoLoss(nn.Module):
    def __init__(self, auto_scale=True):
        super(TopoLoss, self).__init__()
        self.auto_scale = auto_scale
    @timeit
    def forward(self, inputs, target):
        persistence_layer = ttnn.CubicalComplex()
        before_persistence_diagram = persistence_layer(inputs)
        after_persistence_diagram = persistence_layer(target)
        distance = ttnn.WassersteinDistance()(before_persistence_diagram[0][0], after_persistence_diagram[0][0])
        # scalar
        if self.auto_scale:
            while distance > 2:
                distance = distance / 10
        return distance

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


class Loss1(nn.Module):
    def __init__(self) -> None:
        super(Loss1, self).__init__()
        self.alpha = .5
        self.beta = 1.
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.DiceLoss = DiceLoss()

    def forward(self, inputs, targets):
        return self.alpha * self.BCELoss(inputs, targets) + \
                self.beta * self.DiceLoss(inputs, targets)


class Loss2(nn.Module):
    def __init__(self):
        super(Loss2, self).__init__()

        # hyperparameters
        self.alpha = .5
        self.beta = 1.
        self.gamma = .5

        # Loss
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.DiceLoss = DiceLoss()
        self.TopoLoss = TopoLoss()

    def forward(self, inputs, targets):
        return self.alpha * self.BCELoss(inputs, targets) + \
                self.beta * self.DiceLoss(inputs, targets) + \
                self.gamma * self.TopoLoss(inputs, targets)


if __name__ == '__main__':
    reader = usual_reader()
    # shape = (633, 259, 352)
    # seg, _, _, _ = reader(r"D:\Work\data\label.nii.gz")
    # label, _, _, _ = reader(r"D:\Work\data\label.nii.gz")
    rand = torch.randn(1, 1, 100, 100)
    seg = reader(r'D:\data\FAZ\Domain1\test\mask\001_D_1.png', 'torch').to(torch.float32)
    label = reader(r'D:\data\FAZ\Domain1\test\mask\065_M_59.png', 'torch').to(torch.float32)
    dis = TopoLoss()(seg, label)
    print(dis)