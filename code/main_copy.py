import numpy as np
import torch
import argparse

from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms
from data.FAZ import FAZDataset
from medpy.metric.binary import dc

from models.UNet import UNet
from models.distance_loss import Distance_Loss_copy as Loss1

from utils.common_utils import set_seed, get_config
from utils.evaluate import hd95, assd
from yaoxin_tools.tools import getLocalTime, writer_log, usual_reader
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


if __name__ == '__main__':
    args = get_config()

    set_seed(77)
    # Transform for preprocessing images
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    # Create datasets and dataloaders
    train_dataset = FAZDataset(root_dir=r'../FAZ', mode='train', transform=transform)
    test_dataset = FAZDataset(root_dir=r'../FAZ', mode='test', transform=transform)
    # train_dataset = FAZDataset(root_dir=args.data_dir, mode='train', transform=transform)
    # test_dataset = FAZDataset(root_dir=args.data_dir, mode='test', transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4)

    # from yaoxin_tools.template import check_bestNumWorkers
    # numworker = check_bestNumWorkers(train_loader)
    # exit()

    # Define training loop
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

    model = UNet(1, 1).to(device)
    if args.resume:
        model.load_state_dict(torch.load(args.resume))
        args.resume = 1
    else:
        model.apply(init_weights)
    model.to(device)
    logger = writer_log(f'./logs/{getLocalTime()}', **vars(args))
    logger(f"Length of Training Data:{len(train_dataset)}, Length of Testing Data:{len(test_dataset)}")
    criterion1 = Loss1()  # Using binary cross entropy loss for segmentation
    # criterion2= Loss2()
    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    logger(f'start running at device: {device}')
    # Training loop
    num_epochs = args.epochs
    test_best = {'dice': 0, 'hd95': 0, 'assd': 0, 'ap':0, 'epoch': 0}
    for epoch in range(num_epochs):
        model.train()
        epoch_loss1 = 0
        epoch_loss2 = 0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion1(outputs, masks)  # Calculate loss using binary cross entropy
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss1 += loss.item()

        avg_loss1 = epoch_loss1 / len(train_loader)
        logger(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss1:.4f}")
        # logger(f"Epoch [{epoch + 1}/{num_epochs}], Loss2: {avg_loss2:.4f}")

        if (epoch+1) % 1 == 0:
            model.eval()
            with torch.no_grad():
                dice_scores, hd95_scores, assd_scores = [], [], []
                ap = 1e-9
                for images, masks in test_loader:
                    images, masks = images.to(device), masks.to(device)
                    outputs = model(images)
                    outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities
                    preds = (outputs > 0.5).float()  # Binarize the output

                    preds, masks = preds.cpu().numpy(), masks.cpu().numpy()
                    # Dice coefficient calculation
                    dice_score, hd_res, assd_res = dc(preds, masks), hd95(preds, masks), assd(preds, masks)
                    dice_scores.append(dice_score)
                    hd95_scores.append(hd_res)
                    assd_scores.append(assd_res)
            dice_s = np.mean(dice_scores)
            assd_s = np.mean(assd_scores)
            hd95_s = np.mean(hd95_scores)
            ap =  dice_s - .1 *  - .1 * assd_s
            
            if ap > test_best['ap']:
                test_best['dice'] = np.mean(dice_scores)
                test_best['hd95'] = np.mean(hd95_scores)
                test_best['assd'] = np.mean(assd_scores)
                test_best['ap'] = ap
                test_best['epoch'] = epoch+1
                if args.resume:
                    torch.save(model.state_dict(), f'./checkpoints/UNet_best_lr_{args.lr}_bs_{args.batch_size}_resume.pth')
                else:
                    torch.save(model.state_dict(), f'./checkpoints/UNet_best_lr_{args.lr}_bs_{args.batch_size}.pth')
            torch.save(model.state_dict(), f'./checkpoints/UNet_epoch{epoch+1}_lr_{args.lr}_bs_{args.batch_size}.pth')
            logger.flag = True
            logger(f"Epoch{epoch+1}: Average Dice Score: {dice_s:.4f}")
            logger(f"Epoch{epoch+1}: Average HD95: {assd_s:.4f}")
            logger(f"Epoch{epoch+1}: Average ASSD: {hd95_s:.4f}")
            logger(f"Epoch{epoch+1}: Average AP: {ap}")
            logger(f"Current Best Testing Info: Best Dice Score: {test_best['dice']:.4f}, with HD95 = {test_best['hd95']} and ASSD = {test_best['assd']} and AP = {test_best['ap']} at epoch {test_best['epoch']+1}")
    # Testing loop
    model.eval()
    dice_scores, hd95_scores, assd_scores = [], [], []

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            # outputs = torch.sigmoid(outputs)  # Apply sigmoid to convert logits to probabilities
            preds = (outputs > 0.5).float()  # Binarize the output

            preds, masks = preds.cpu().numpy(), masks.cpu().numpy()
            # Dice coefficient calculation
            dice_score, hd_res, assd_res = dc(preds, masks), hd95(preds, masks), assd(preds, masks)
            dice_scores.append(dice_score)
            hd95_scores.append(hd_res)
            assd_scores.append(assd_res)
            dice_s = np.mean(dice_scores)
            assd_s = np.mean(assd_scores)
            hd95_s = np.mean(hd95_scores)
            ap =  dice_s - .1 *  - .1 * assd_s
            
            if ap > test_best['ap']:
                test_best['dice'] = np.mean(dice_scores)
                test_best['hd95'] = np.mean(hd95_scores)
                test_best['assd'] = np.mean(assd_scores)
                test_best['ap'] = ap
                test_best['epoch'] = epoch+1
                if args.resume:
                    torch.save(model.state_dict(), f'./checkpoints/UNet_best_lr_{args.lr}_bs_{args.batch_size}_resume.pth')
                else:
                    torch.save(model.state_dict(), f'./checkpoints/UNet_best_lr_{args.lr}_bs_{args.batch_size}.pth')
        logger(f"Epoch{epoch+1}: Average Dice Score: {dice_s:.4f}")
        logger(f"Epoch{epoch+1}: Average HD95: {assd_s:.4f}")
        logger(f"Epoch{epoch+1}: Average ASSD: {hd95_s:.4f}")
        logger(f"Epoch{epoch+1}: Average AP: {ap}")
        logger(f"Current Best Testing Info: Best Dice Score: {test_best['dice']:.4f}, with HD95 = {test_best['hd95']} and ASSD = {test_best['assd']} and AP = {test_best['ap']} at epoch {test_best['epoch']+1}")