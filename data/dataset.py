import numpy as np
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, dataset_path):
        data = np.load(dataset_path)
        self.coordiates = data['X']
        self.values = data['Y']
        self.input_dim = self.coordiates.shape[1]

    def __len__(self):
        return len(self.coordiates)

    def __getitem__(self, idx):
        return self.coordiates[idx], self.values[idx]
