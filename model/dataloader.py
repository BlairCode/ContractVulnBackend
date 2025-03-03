# This file handles:
# dataset building 
# & dataset loading

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset

from .w2v.model import get_embd
from .w2v import epoch2cp
from .w2v import w2v_save_path

# After preprocessing, we have the following structure:
# out/
# ├── kind/
# │   ├── class/
# │   │   ├── filename/
# │   │   │   ├── antlr.txt
# │   │   │   └── sliced.txt

# Do note that if no sliced.txt or antlr.txt is found, 
# the file is skipped

# We intend to build a dataset, where each entry is like such:
# (label, tensor)
# where label is 1 if the file has a vulnerability, 0 otherwise
# With such a dataset and leveraging PyTorch's DataLoader,
# we can provide batches of data to the model with ease

supported_kinds = ['ree']
good = 'non_vul'
bad = 'vul'

def get_data(w2v_cp):
    data = []
    for kind in supported_kinds:
        kind_path = os.path.join('out', kind)
        if not os.path.exists(kind_path):
            print(f"Supported kind {kind} not found, verify out directory structure,"
                   "and edit supported_kinds in model/dataloader.py")
            continue

        good_path = os.path.join(kind_path, good)
        bad_path = os.path.join(kind_path, bad)

        # have vulnerabilities
        for entry_name in os.listdir(bad_path):
            single_entry_path = os.path.join(bad_path, entry_name)
            files = os.listdir(single_entry_path)

            # assert that both files are present
            if not 'sliced.txt' in files:
                continue
            if not 'antlr.txt' in files:
                continue

            code = ''
            sliced_path = os.path.join(single_entry_path, 'sliced.txt')
            with open(sliced_path, 'r') as f:
                s_code = f.read().replace('\n', ' ')
                code += s_code
            code += ' '
            antlr_path = os.path.join(single_entry_path, 'antlr.txt')
            with open(antlr_path, 'r') as f:
                a_code = f.read().replace('\n', ' ')
                code += a_code

            # w2v_cp_path = os.path.join(w2v_save_path, epoch2cp(w2v_cp))
            vecs = get_embd(code, w2v_cp)
            data.append((1, vecs))

        # no vulnerabilities
        for entry_name in os.listdir(good_path):
            single_entry_path = os.path.join(good_path, entry_name)
            files = os.listdir(single_entry_path)

            # assert that both files are present
            if not 'sliced.txt' in files:
                continue
            if not 'antlr.txt' in files:
                continue

            code = ''
            sliced_path = os.path.join(single_entry_path, 'sliced.txt')
            with open(sliced_path, 'r') as f:
                s_code = f.read().replace('\n', ' ')
                code += s_code
            code += ' '
            antlr_path = os.path.join(single_entry_path, 'antlr.txt')
            with open(antlr_path, 'r') as f:
                a_code = f.read().replace('\n', ' ')
                code += a_code

            # w2v_cp_path = os.path.join(w2v_save_path, epoch2cp(w2v_cp))
            vecs = get_embd(code, w2v_cp)
            data.append((0, vecs))
    return data

class CodeDataset(Dataset):
    def __init__(self, w2v_cp, device, mode='train'):
        self.root_path = os.path.join(os.path.curdir, 'out')

        data = get_data(w2v_cp)

        # now data is a list of tuples like:
        # [(label1, tensor1), ...]
        labels = []
        sequences = []
        for label, vec in data:
            labels.append(label)
            sequences.append(vec.detach())
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0.0).to(device)

        # TODO: fixed mx ctx length
        positive_mask = (labels == 1)
        positive_idxs = torch.where(positive_mask)[0]
        negative_idxs = torch.where(~positive_mask)[0]
        if len(positive_idxs) > len(negative_idxs):
            undersampled_negative_idxs = torch.randperm(
                len(negative_idxs), 
                device=device
            )[:len(positive_idxs)]
            resampled_idxs = torch.cat([positive_idxs, negative_idxs[undersampled_negative_idxs]])
        else:
            undersampled_positive_idxs = torch.randperm(
                len(positive_idxs),
                device=device
            )[:len(negative_idxs)]
            resampled_idxs = torch.cat([positive_idxs[undersampled_positive_idxs], negative_idxs])
        x_resampled = padded_sequences[resampled_idxs]
        y_resampled = labels[resampled_idxs]

        dataset = TensorDataset(x_resampled, y_resampled)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        # Warning: keep the seed so that the split is deterministic
        torch.manual_seed(42)
        self.train_dataset, self.test_dataset = random_split(dataset, [train_size, test_size])

        if mode == 'train':
            self.data = self.train_dataset
        else:
            self.data = self.test_dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
