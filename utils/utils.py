import random

import numpy as np
import torch

def save_checkpoint(model, checkpoint_dir):
    model.save_pretrained(checkpoint_dir)

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)