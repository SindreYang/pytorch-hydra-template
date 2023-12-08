import logging
import random
import numpy as np
import torch
import os


def seed_torch(seed=1024):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)  # 为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果您使用的是多GPU。
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_logger(name=__name__) -> logging.Logger:
    """python命令行记录器"""

    logger = logging.getLogger(name)
    return logger


