import torch as th
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler

from syft.generic import ObjectStorage
from syft.federated.train_config import TrainConfig


class FederatedClient(ObjectStorage):
    """A Client able to execute federated learning in local datasets."""

    def __init__(self, datasets=None):
        super().__init__()
        self.datasets = datasets if datasets is not None else dict()
        self.optimizer = None
        self.train_config = None

    def add_dataset(self, dataset, key: str):
        self.datasets[key] = dataset

    def remove_dataset(self, key: str):
        if key in self.datasets:
            del self.datasets[key]

    def set_obj(self, obj: object):
        """Registers objects checking if which objects it should cache.

        Args:
            obj: An object to be registered.
        """
        if isinstance(obj, TrainConfig):
            self.train_config = obj
            self.optimizer = None
        else:
            super().set_obj(obj)

    def _build_optimizer(
        self, optimizer_name: str, model, lr: float, weight_decay: float
    ) -> th.optim.Optimizer:
        """Build an optimizer if needed.

        Args:
            optimizer_name: A string indicating the optimizer name.
            lr: A float indicating the learning rate.
            weight_decay: Weight decay parameter of the optimizer
        Returns:
            A Torch Optimizer.
        """
        if self.optimizer is not None:
            return self.optimizer

        optimizer_name = optimizer_name.lower()
        if optimizer_name == "sgd":
            optim_args = dict()
            optim_args["lr"] = lr
            if weight_decay is not None:
                optim_args["weight_decay"] = weight_decay
            self.optimizer = th.optim.SGD(model.parameters(), **optim_args)
        else:
            raise ValueError("Unknown optimizer: {}".format(optimizer_name))
        return self.optimizer

    def fit(self, dataset_key: str, **kwargs):
        """Fits a model on the local dataset as specified in the local TrainConfig object.

        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            **kwargs: Unused.

        Returns:
            loss: Training loss on the last batch of training data.
        """
        if self.train_config is None:
            raise ValueError("TrainConfig not defined.")

        model = self.get_obj(self.train_config._model_id).obj
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj

        self._build_optimizer(
            self.train_config.optimizer,
            model,
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )

        return self._fit(model=model, dataset_key=dataset_key, loss_fn=loss_fn)

    def _create_data_loader(self, dataset_key: str, shuffle: bool = False):
        data_range = range(len(self.datasets[dataset_key]))
        if shuffle:
            sampler = RandomSampler(data_range)
        else:
            sampler = SequentialSampler(data_range)
        data_loader = th.utils.data.DataLoader(
            self.datasets[dataset_key],
            batch_size=self.train_config.batch_size,
            sampler=sampler,
            num_workers=0,
        )
        return data_loader

    def _fit(self, model, dataset_key, loss_fn):
        model.train()
        data_loader = self._create_data_loader(
            dataset_key=dataset_key, shuffle=self.train_config.shuffle
        )

        loss = None
        iteration_count = 0
        for (data, target) in data_loader:
            # Set gradients to zero
            self.optimizer.zero_grad()

            # Update model
            output = model(data)
            loss = loss_fn(target=target, pred=output)
            loss.backward()
            self.optimizer.step()

            # Update and check interation count
            iteration_count += 1
            if iteration_count >= self.train_config.max_nr_batches >= 0:
                break

        return loss
