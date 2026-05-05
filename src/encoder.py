"""Dense Encoder models."""
import torch
import torch.nn as nn

class BiEncoder(nn.Module):
    def __init__(self, model_name='bert-base-uncased'):
        super().__init__()
