import torch

from models.UNet import UNet
from utils.visualize import plot_2d_results

if __name__ == '__main__':
    state_dict = torch.load(r'D:/Course/Signal Processing/code_task1/checkpoints/UNet_best_lr_0.02_bs_48.pth')
    model = UNet(1,1)
    model.load_state_dict(state_dict)
    plot_2d_results(model, r'E:/Datasets/FAZ/data/FAZ/Domain1/test/imgs/', 
                    gt=r'E:/Datasets/FAZ/data/FAZ/Domain1/test/mask/', 
                    out='./vis_output/results.png', show_num=6, shuffle=True, cmap='gray', 
                    show=False, binary_thre=0, fig_size=(5,10))