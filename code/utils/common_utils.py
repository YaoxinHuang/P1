import random
import numpy as np
import torch
from torch.backends import cudnn
import logging, argparse


def set_seed(seed: int = 42):
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', '--bs', default=16, type=int, help='batch size of each gpu')
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--cuda', default=0, type=int, help='GPU id')
    parser.add_argument('--lr', default=0.02, type=float, help='learning rate')
    parser.add_argument('--resume', default=0, help='resume train or not')
    # parser.add_argument('--data_dir', '--data', type=str, default='../FAZ/', help='data dir')
    parser.add_argument('--sr', '--shift_ratio', default=0.3, type=float, help='ratio of shift domain in training')
    
    return parser.parse_args()


def set_logger(log_path):
    """
    配置log
    :param log_path:s
    :return:
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # 由于每调用一次set_logger函数，就会创建一个handler，会造成重复打印的问题，因此需要判断root logger中是否已有该handler
    if not any(handler.__class__ == logging.FileHandler for handler in logger.handlers):
        file_handler = logging.FileHandler(log_path)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(lineno)d - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if not any(handler.__class__ == logging.StreamHandler for handler in logger.handlers):
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)