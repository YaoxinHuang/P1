import medpy
import numpy as np

def hd95(s, g):
    if np.sum(s) == 0:
        return 0
    return medpy.metric.binary.hd95(s, g, voxelspacing=None)


def assd(s, g):
    if np.sum(s) == 0:
        return 0
    return medpy.metric.binary.assd(s, g, voxelspacing=None)