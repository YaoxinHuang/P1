import numpy as np
import torch
import argparse

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.ODOC import ODOCDataset
from medpy.metric.binary import dc

from models.UNet import UNet
from models.Loss import Loss1, Loss2, distanceWeighted_Loss, Loss

from utils.common_utils import set_seed, get_config
from utils.evaluate import hd95, assd
from yaoxin_tools.tools import getLocalTime, writer_log, usual_reader, timeit
import torch.nn.init as init

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

@timeit
def test(model, loader):
    model.eval()
    od_dice_scores, od_hd95_scores, od_assd_scores = [], [], []
    oc_dice_scores, oc_hd95_scores, oc_assd_scores = [], [], []
    test_loader = loader
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            outputs = (torch.sigmoid(outputs) > 0.5).float()  # Apply sigmoid to convert logits to probabilities
            preds, masks = outputs.cpu().numpy(), masks.cpu().numpy()
            # Dice coefficient calculation
            od_dice_score, od_hd_res, od_assd_res = dc(preds[:, 1:2], masks[:, 1:2]), hd95(preds[:, 1:2], masks[:, 1:2]), assd(preds[:, 1:2], masks[:, 1:2])
            oc_dice_score, oc_hd_res, oc_assd_res = dc(preds[:, 2:3], masks[:, 2:3]), hd95(preds[:, 2:3], masks[:, 2:3]), assd(preds[:, 2:3], masks[:, 2:3])
            
            od_dice_scores.append(od_dice_score)
            od_hd95_scores.append(od_hd_res)
            od_assd_scores.append(od_assd_res)
            oc_dice_scores.append(oc_dice_score)
            oc_hd95_scores.append(oc_hd_res)
            oc_assd_scores.append(oc_assd_res)

            od_dice_s = np.mean(od_dice_scores)
            od_assd_s = np.mean(od_assd_scores)
            od_hd95_s = np.mean(od_hd95_scores)
            oc_dice_s = np.mean(oc_dice_scores)
            oc_assd_s = np.mean(oc_assd_scores)
            oc_hd95_s = np.mean(oc_hd95_scores)
            
            mean_dice_s = (od_dice_s + oc_dice_s) / 2
            mean_assd_s = (od_assd_s + oc_assd_s) / 2
            mean_hd95_s = (od_hd95_s + oc_hd95_s) / 2
            
            if mean_dice_s > test_best['dice']:
                test_best['dice'] = mean_dice_s
                test_best['hd95'] = mean_hd95_s
                test_best['assd'] = mean_assd_s
                # test_best['ap'] = ap
                test_best['epoch'] = epoch+1
                if args.resume:
                    torch.save(model.state_dict(), f'./checkpoints/UNet_best_lr_{args.lr}_bs_{args.batch_size}_resume.pth')
                else:
                    torch.save(model.state_dict(), f'./checkpoints/UNet_best_lr_{args.lr}_bs_{args.batch_size}.pth')
        logger(f"Epoch{epoch+1}: Average Dice Score: OD:{od_dice_s:.4f}, OC:{oc_dice_s:.4f}")
        logger(f"Epoch{epoch+1}: Average HD95: OD:{od_hd95_s:.4f}, OC:{oc_hd95_s:.4f}")
        logger(f"Epoch{epoch+1}: Average ASSD: OD:{od_assd_s:.4f}, OC:{oc_assd_s:.4f}")
        # logger(f"Epoch{epoch+1}: Average AP: {ap}")
        logger(f"Current Best Testing Info: Best Dice Score: {test_best['dice']:.4f}, with HD95 = {test_best['hd95']} and ASSD = {test_best['assd']} at epoch {test_best['epoch']}")

if __name__ == '__main__':
    args = get_config()

    set_seed(77)
    # Transform for preprocessing images
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets and dataloaders
    train_dataset = ODOCDataset(root_dir=r'../ODOC', mode='train', transform=transform, alpha=0.05)
    test_dataset = ODOCDataset(root_dir=r'../ODOC', mode='test', transform=transform, alpha=0.05)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    # from yaoxin_tools.template import check_bestNumWorkers
    # numworker = check_bestNumWorkers(train_loader)
    # exit()

    # Define training loop
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    model = UNet(3, 3).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        args.resume = 1
    else:
        model.apply(init_weights)
    model.to(device)
    logger = writer_log(f'./logs/{getLocalTime()}', **vars(args))
    logger(f"Length of Training Data:{len(train_dataset)}, Length of Testing Data:{len(test_dataset)}")
    criterion1 = Loss()  # Using binary cross entropy loss for segmentation
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    logger(f'start running at device: {device}')
    # Training loop
    num_epochs = args.epochs
    test_best = {'dice': 0, 'hd95': 0, 'assd': 0, 'epoch': 0}
    for epoch in range(num_epochs):
        model.train()
        epoch_loss_1 = 0
        epoch_loss_2 = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            # Forward pass
            outputs = model(images)
            loss = criterion1(outputs, masks)  # Calculate loss using binary cross entropy
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss_1 += loss.item()
            # epoch_loss_2 += diceLoss.item()

        avg_loss_1 = epoch_loss_1 / len(train_loader)
        avg_loss_2 = epoch_loss_2 / len(train_loader)

        logger(f"Epoch [{epoch + 1}/{num_epochs}], weightedLoss: {avg_loss_1:.4f}")
        # logger(f"Epoch [{epoch + 1}/{num_epochs}], DiceLoss: {avg_loss_2:.4f}")

        if (epoch+1) % 50 == 0 or (epoch+1) == args.epochs:    
            # Testing loop
            test(model, test_loader)
    test(model, test_loader)