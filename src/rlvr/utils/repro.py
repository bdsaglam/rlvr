import random


def set_seed(s, reproducible=False):
    "Set random seed for `random`, `torch`, and `numpy` (where available)"
    random.seed(s)
    
    try:
        import numpy as np

        np.random.seed(s % (2**32 - 1))
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        if reproducible:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
