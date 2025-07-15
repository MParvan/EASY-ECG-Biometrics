# -*- coding: utf-8 -*-
"""
Created on Mon Jul  7 12:08:31 2025

@author: milad
"""

import os
import numpy as np
from torch.utils.data import Dataset

class ECGIDDataset(Dataset):
    def __init__(self, path, task='identification', segment_fn=None, transform=None):
        self.task = task
        self.segment_fn = segment_fn
        self.transform = transform
        self.samples, self.labels = self._load(path)

    def _load(self, path):
        samples, labels = [], []
        for file in sorted(os.listdir(path)):
            print(file)
            if file.endswith(".npy"):
                label = int(file.split("_")[1].split(".")[0])
                signal = np.load(os.path.join(path, file))
                beats = self.segment_fn(signal) if self.segment_fn else [signal]
                samples += beats
                labels += [label] * len(beats)
        return samples, labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]
        if self.transform:
            x = self.transform(x)
        return x.astype(np.float32), self.labels[idx]
