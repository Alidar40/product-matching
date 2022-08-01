import random

import numpy as np
import torch
from torch.utils.data import Dataset


class ProductSet(Dataset):
    def __init__(self, data_frame, mode="single"):
        self.df = data_frame.copy()
        self.mode = mode

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        if self.mode == "single":
            return torch.as_tensor(self.df.iloc[idx]["embs"]), torch.as_tensor(self.df.iloc[idx]["label_group"], dtype=torch.int32)
        elif self.mode == "pair":
            anchor = self.df.iloc[idx]
            if random.random() >= 0.5:
                positive = self.df[self.df["label_group"] == anchor["label_group"]].sample().iloc[0]
                return torch.from_numpy(np.concatenate([anchor["embs"], positive["embs"]])), torch.as_tensor([1])
            else:
                negative = self.df[self.df["label_group"] != anchor["label_group"]].sample().iloc[0]
                return torch.from_numpy(np.concatenate([anchor["embs"], negative["embs"]])), torch.as_tensor([0])
        else:
            raise ValueError("Bad dataset access mode")


class ProductTestSet(Dataset):
    def __init__(self, data_frame):
        self.df = data_frame.copy()

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        return torch.as_tensor(self.df.iloc[idx]["embs"]), torch.zeros((1, 1))
