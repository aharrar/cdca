import numpy as np
import torch
from imblearn.over_sampling import RandomOverSampler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sda_utils import get_one_hot_encoding


class CdcaDataset(Dataset):
    def __init__(self, data, labels, batch_id, class_number=None, method=None):
        self.data = data
        self.labels = labels
        self.batch_id = batch_id

        self.class_number = class_number
        if method == "resample":
            self.src_data, self.src_labels = RandomOverSampler().fit_resample(self.src_data,
                                                                              self.src_labels)

    def __len__(self):
        # return self.size
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # if self.class_number!=None:
        #     return self.data[idx], get_one_hot_encoding(self.src_labels[idx], self.class_number)

        return self.data[idx], self.labels[idx], self.batch_id[idx]


class TargetDataset(Dataset):
    def __init__(self, target_data, target_labels, class_number=None, method=None):
        self.target_data = target_data
        self.target_labels = target_labels
        self.class_number = class_number

        if method == "resample":
            self.target_data, self.target_labels = RandomOverSampler().fit_resample(self.target_data,
                                                                                    self.target_labels)

    def __len__(self):
        # return self.size
        return len(self.target_data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.class_number != None:
            return self.target_data[idx], get_one_hot_encoding(self.target_labels[idx], self.class_number)

        return self.target_data[idx], self.target_labels[idx]


class ForeverDataIterator:
    r"""A data iterator that will never stop producing data"""

    def __init__(self, data_loader: DataLoader, device=None):
        self.data_loader = data_loader
        self.iter = iter(self.data_loader)
        self.device = device

    def __next__(self):
        try:
            data = next(self.iter)

        except StopIteration:
            self.iter = iter(self.data_loader)
            data = next(self.iter)

        return data

    def __len__(self):
        return len(self.data_loader)
