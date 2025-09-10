import torch
from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, dfs, sample_interval, sequence_length, transform=None):
        if not isinstance(dfs, list):
            dfs = [dfs]

        self.sequence_sample_count = int(sequence_length / sample_interval)
        self.sequences = []

        for df in dfs:
            for start in range(0, len(df) - self.sequence_sample_count + 1, self.sequence_sample_count):
                end = start + self.sequence_sample_count
                self.sequences.append(df.iloc[start:end].to_numpy())

        self.transform = transform

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        x, y = seq.clone(), seq.clone()

        if self.transform:
            x, y = self.transform(x, y)

        return x, y


def masked_transform(mask_prob=0.3, mask_value=0.0):
    def transform(x, y):
        mask = torch.rand(x.shape) < mask_prob
        x = torch.where(mask, torch.full_like(x, mask_value), x)
        return x, y
    return transform


def identity_transform():
    def transform(x, y):
        return x, y
    return transform
