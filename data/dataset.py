import os

import numpy as np
from jax import numpy as jnp
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, example_name):
        """Initialize the dataset with the given example name.

        Args:
            example_name (str): The name of the example dataset to load.
        Raises:
            FileNotFoundError: If the dataset files are not found.
        """
        data = np.load(os.path.join('input', f'{example_name}.npy'))
        self.X = jnp.array(data['X'])
        self.Y = jnp.array(data['Y'])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
