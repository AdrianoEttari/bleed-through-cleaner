#%%
import torch
import numpy as np


    
    
def masked_l1(pred, gt, hole_mask):
    return torch.mean(
        torch.abs(pred - gt) * hole_mask.unsqueeze(1)
    )