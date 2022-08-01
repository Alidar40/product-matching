import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, emb_size=300):
        super().__init__()

        self.emb_size = emb_size

        self.transformations = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.ReLU(),
            nn.BatchNorm1d(emb_size),

            nn.Linear(emb_size, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),

            nn.Linear(128, 12),
            nn.ReLU(),
            nn.BatchNorm1d(12),
        )

    def forward(self, x):
        return self.transformations(x)
