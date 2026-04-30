"""Contrastive loss functions."""
import torch
import torch.nn as nn

class InfoNCELoss(nn.Module):
    def __init__(self, tau=0.05):
        super().__init__()
        self.tau = tau
    # temperature scaling
    # numerical stability
