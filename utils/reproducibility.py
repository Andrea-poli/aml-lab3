"""Fix all random seeds so that a run can be reproduced (Lab01, exercises 4-5)."""
import random

import numpy as np
import torch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)           # CPU and all GPUs
    torch.cuda.manual_seed_all(seed)  # explicit, for clarity
