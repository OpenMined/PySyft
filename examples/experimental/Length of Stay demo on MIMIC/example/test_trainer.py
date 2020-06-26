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

grand_parent_folder = os.path.dirname(os.path.dirname(__file__))
CONFIG_PATH = os.path.join(grand_parent_folder, "metadata", "config_result_paper.json")


def test_trainer() -> None:
    """function showing how to use the train function
    """
    sys.path.append(".")
    path_dirname = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(path_dirname)

    from pytorch import train
    from pytorch import Arguments
    from pytorch.trainer import set_seed

    n_features = 10
    n_samples = 1000
    set_seed(1)
    X = torch.randint(
        10, size=(n_samples, n_features), dtype=torch.float
    )  # (n_samples, n_features)
    Y = torch.randint(3, size=(n_samples, 1), dtype=torch.float)  # (n_samples,)

    dataset = TensorDataset(X, Y)

    with open(CONFIG_PATH) as json_file:
        train_params = json.load(json_file)

    train_dataloader, val_dataloader, _ = train_val_test_dataloaders(
        dataset, batch_size=train_params.pop("batch_size")
    )
    train_params.update(
        {
            "train_dataloader": train_dataloader,
            "val_dataloader": val_dataloader,
            "n_features": n_features,
            "nb_epoch": 2,
        }
    )

    args = Arguments()
    args.from_dict(train_params)

    logging.info("The train function is tested with the following parameters:")
    logging.info(pformat(train_params))
    model = train(
        args=args,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        n_features=n_features,
        final_activation=args.final_activation,
        loss_name=args.loss_name,
        nb_epoch=args.nb_epoch,
        learning_rate=args.learning_rate,
        ffn_depth=args.ffn_depth,
        optim_name=args.optim_name,
    )

    assert isinstance(model, nn.Module)


def train_val_test_dataloaders(
    dataset: object, batch_size: int, val_frac: float = 0.2, test_frac: float = 0.2, seed: int = 1,
) -> (object, object, object):
    """Function that creates 3 DataLoaders for train, validation, test sets
        respectively
    """
    dataset_size = len(dataset)
    lenghts = []
    lenghts.append(int(np.floor(val_frac * dataset_size)))
    lenghts.append(int(np.floor(test_frac * dataset_size)))
    lenghts.insert(0, dataset_size - lenghts[0] - lenghts[1])
    assert lenghts[2] >= 0

    datasets = torch.utils.data.random_split(dataset, lenghts)
    logging.debug(f"datasets lenghts: {[len(data) for data in datasets]}")

    dataloaders = [
        DataLoader(dataset, batch_size=batch_size, drop_last=True) for dataset in datasets
    ]
    logging.debug(f"dataloaders lenghts: {[len(data) for data in dataloaders]}")
    return dataloaders  # [train_dataloader, validation_dataloader, test_dataloader]


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    test_trainer()
