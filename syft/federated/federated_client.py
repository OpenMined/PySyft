import torch as th
from torch.utils.data import BatchSampler, RandomSampler, SequentialSampler
import numpy as np

from syft.generic.object_storage import ObjectStorage
from syft.federated.train_config import TrainConfig


class FederatedClient(ObjectStorage):
    """A Client able to execute federated learning in local datasets."""

    def __init__(self, datasets=None):
        super().__init__()
        self.datasets = datasets if datasets is not None else dict()
        self.optimizer = None
        self.train_config = None

    def add_dataset(self, dataset, key: str):
        if key not in self.datasets:
            self.datasets[key] = dataset
        else:
            raise ValueError(f"Key {key} already exists in Datasets")

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

    def _check_train_config(self):
        if self.train_config is None:
            raise ValueError("Operation needs TrainConfig object to be set.")

    def _build_optimizer(
        self, optimizer_name: str, model, optimizer_args: dict
    ) -> th.optim.Optimizer:
        """Build an optimizer if needed.

        Args:
            optimizer_name: A string indicating the optimizer name.
            optimizer_args: A dict containing the args used to initialize the optimizer.
        Returns:
            A Torch Optimizer.
        """
        if self.optimizer is not None:
            return self.optimizer

        if optimizer_name in dir(th.optim):
            optimizer = getattr(th.optim, optimizer_name)
            self.optimizer = optimizer(model.parameters(), **optimizer_args)
        else:
            raise ValueError("Unknown optimizer: {}".format(optimizer_name))
        return self.optimizer

    def fit(self, dataset_key: str, device: str = "cpu", **kwargs):
        """Fits a model on the local dataset as specified in the local TrainConfig object.

        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            **kwargs: Unused.

        Returns:
            loss: Training loss on the last batch of training data.
        """
        self._check_train_config()

        if dataset_key not in self.datasets:
            raise ValueError("Dataset {} unknown.".format(dataset_key))

        model = self.get_obj(self.train_config._model_id).obj
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj

        self._build_optimizer(
            self.train_config.optimizer, model, optimizer_args=self.train_config.optimizer_args
        )

        return self._fit(model=model, dataset_key=dataset_key, loss_fn=loss_fn, device=device)

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

    def _fit(self, model, dataset_key, loss_fn, device="cpu"):
        model.train()
        data_loader = self._create_data_loader(
            dataset_key=dataset_key, shuffle=self.train_config.shuffle
        )

        loss = None
        iteration_count = 0

        for _ in range(self.train_config.epochs):
            for (data, target) in data_loader:
                # Set gradients to zero
                self.optimizer.zero_grad()

                # Update model
                output = model(data.to(device))
                loss = loss_fn(target=target.to(device), pred=output)
                loss.backward()
                self.optimizer.step()

                # Update and check interation count
                iteration_count += 1
                if iteration_count >= self.train_config.max_nr_batches >= 0:
                    break

        return loss

    def evaluate(
        self,
        dataset_key: str,
        return_histograms: bool = False,
        nr_bins: int = -1,
        return_loss: bool = True,
        return_raw_accuracy: bool = True,
        device: str = "cpu",
    ):
        """Evaluates a model on the local dataset as specified in the local TrainConfig object.

        Args:
            dataset_key: Identifier of the local dataset that shall be used for training.
            return_histograms: If True, calculate the histograms of predicted classes.
            nr_bins: Used together with calculate_histograms. Provide the number of classes/bins.
            return_loss: If True, loss is calculated additionally.
            return_raw_accuracy: If True, return nr_correct_predictions and nr_predictions
            device: "cuda" or "cpu"

        Returns:
            Dictionary containing depending on the provided flags:
                * loss: avg loss on data set, None if not calculated.
                * nr_correct_predictions: number of correct predictions.
                * nr_predictions: total number of predictions.
                * histogram_predictions: histogram of predictions.
                * histogram_target: histogram of target values in the dataset.
        """
        self._check_train_config()

        if dataset_key not in self.datasets:
            raise ValueError("Dataset {} unknown.".format(dataset_key))

        eval_result = dict()
        model = self.get_obj(self.train_config._model_id).obj
        loss_fn = self.get_obj(self.train_config._loss_fn_id).obj
        model.eval()
        device = "cuda" if device == "cuda" else "cpu"
        data_loader = self._create_data_loader(dataset_key=dataset_key, shuffle=False)
        test_loss = 0.0
        correct = 0
        if return_histograms:
            hist_target = np.zeros(nr_bins)
            hist_pred = np.zeros(nr_bins)

        with th.no_grad():
            for data, target in data_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                if return_loss:
                    test_loss += loss_fn(output, target).item()  # sum up batch loss
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                if return_histograms:
                    hist, _ = np.histogram(target, bins=nr_bins, range=(0, nr_bins))
                    hist_target += hist
                    hist, _ = np.histogram(pred, bins=nr_bins, range=(0, nr_bins))
                    hist_pred += hist
                if return_raw_accuracy:
                    correct += pred.eq(target.view_as(pred)).sum().item()

        if return_loss:
            test_loss /= len(data_loader.dataset)
            eval_result["loss"] = test_loss
        if return_raw_accuracy:
            eval_result["nr_correct_predictions"] = correct
            eval_result["nr_predictions"] = len(data_loader.dataset)
        if return_histograms:
            eval_result["histogram_predictions"] = hist_pred
            eval_result["histogram_target"] = hist_target

        return eval_result
