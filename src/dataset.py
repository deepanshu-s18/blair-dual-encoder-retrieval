"""PyTorch dataset for triplet contrastive learning."""
import torch
from torch.utils.data import Dataset

class TripletDataset(Dataset):
    pass

def collate_fn(batch):
    pass
    max_q_len = 64
    max_d_len = 256
    # in-memory cache
