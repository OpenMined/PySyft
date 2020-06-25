"""module containing functions showing how to use the other functions in the
    Length of Stay demo on MIMIC folder
"""
import os
import sys
import json
import numpy as np
import logging
import torch
import torch.nn as nn
from pprint import pformat
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

print(os.path.dirname(os.path.dirname(__file__)))
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "metadata", "config_result_paper.json",
)


def test_trainer() -> None:
    """function showing how to use the train function
    """
    sys.path.append(".")
    path_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(path_dirname)

    from pytorch import train

    n_features = 10
    n_samples = 1000
    X = torch.randint(
        10, size=(n_samples, n_features), dtype=torch.float
    )  # (n_samples, n_features)
    Y = torch.randint(3, size=(n_samples, 1), dtype=torch.float)  # (n_samples,)

    dataset = TensorDataset(X, Y)

    with open(CONFIG_PATH) as json_file:
        train_args = json.load(json_file)

    train_dataloader, val_dataloader, _ = train_val_test_dataloaders(
        dataset, batch_size=train_args.pop("batch_size")
    )
    train_args.update(
        {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "n_features": n_features,
        }
    )
    logging.info("The train function is tested with the following parameters:")
    logging.info(pformat(train_args))
    model = train(**train_args)

    assert isinstance(model, nn.Module)


def train_val_test_dataloaders(
    dataset: object, batch_size: int, val_size: float = 0.2, test_size: float = 0.2, seed: int = 1,
) -> (object, object, object):
    """Function that creates 3 DataLoaders for train, validation and test sets
        respectively
    """
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split_val = int(np.floor(val_size * dataset_size))
    split_test = split_val + int(np.floor(test_size * dataset_size))

    np.random.seed(seed)
    np.random.shuffle(indices)
    val_indices, test_indices, train_indices = (
        indices[:split_val],
        indices[split_val:split_test],
        indices[split_test:],
    )

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=True, sampler=train_sampler
    )

    val_dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=True, sampler=valid_sampler
    )

    test_dataloader = DataLoader(
        dataset, batch_size=batch_size, drop_last=True, sampler=test_sampler
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_trainer()
