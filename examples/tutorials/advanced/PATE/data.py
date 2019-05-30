import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


def load_data(train, batch_size):

    """Helper function used to load the train/test data"""

    loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../data",
            train=train,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    return loader


def split(dataset, batch_size, split=0.2):

    index = 0
    length = len(dataset)

    train_set = []
    val_set = []

    for data, target in dataset:

        if index <= (length * split):

            train_set.append([data, target])

        else:

            val_set.append([data, target])

        index += 1

    return train_set, val_set


class NoisyDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, dataloader, model, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataloader = dataloader
        self.model = model
        self.transform = transform
        self.noisy_data = self.process_data()

    def process_data(self):

        noisy_data = []

        for data, _ in self.dataloader:

            noisy_data.append([data, torch.tensor(self.model(data))])

        return noisy_data

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, idx):

        sample = self.noisy_data[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample
