import torch
from torch.utils.data import Dataset
import numpy as np

class BaseSequenceDataset(Dataset):
    def __init__(self, df, sample_interval, sequence_length):
        self.df = df
        self.sample_interval = sample_interval
        self.sequence_sample_count = int(sequence_length / sample_interval)

        self.sequences = []
        for start in range(0, len(df) - self.sequence_sample_count + 1, self.sequence_sample_count):
            end = start + self.sequence_sample_count
            sequence = df.iloc[start:end].to_numpy()
            self.sequences.append(sequence)

    def __len__(self):
        return len(self.sequences)


class SequenceReconstructionDataset(BaseSequenceDataset):
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        x = torch.tensor(sequence, dtype=torch.float32)
        y = torch.tensor(sequence, dtype=torch.float32)
        return x, y


class RandomMaskedSequenceDataset(BaseSequenceDataset):
    def __init__(self, df, sample_interval, sequence_length, mask_prob=0.3, mask_value=0.0):
        super().__init__(df, sample_interval, sequence_length)
        self.mask_prob = mask_prob
        self.mask_value = mask_value

    def __getitem__(self, idx):
        sequence = self.sequences[idx]

        y = torch.tensor(sequence, dtype=torch.float32)

        x = y.clone()
        mask = torch.rand(x.shape) < self.mask_prob
        x = torch.where(mask, torch.full_like(x, self.mask_value), x)

        return x, y
