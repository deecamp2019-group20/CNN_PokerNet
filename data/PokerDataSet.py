import os
import math
import numpy as np
import torch
from torch.utils.data import Dataset
# from torchvision import transforms


def default_loader(root, index):
    f_index = math.ceil((index + 1) / 500)
    fn_state = os.path.join(root, 'data', 'all_state_%d.npy' % f_index)
    fn_label = os.path.join(root, 'label', 'all_label_%d.npy' % f_index)
    f_state = np.load(fn_state)
    f_label = np.load(fn_label)
    state = torch.from_numpy(f_state[index % 500]).float()
    label = f_label[index % 500]

    return state, label


class PokerDataSet(Dataset):

    def __init__(self, root, size, loader=default_loader):
        self.root = root
        self.size = size
        self.loader = loader

    def __getitem__(self, index):
        state, label = self.loader(self.root, index)
        return state, label

    def __len__(self):
        return self.size
