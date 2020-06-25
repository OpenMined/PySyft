"""
Module containing functions useful for training a pytorch neural network
"""
import numpy as np
import logging
import torch.optim as optim
from torch import no_grad
from torch import manual_seed as torch_manual_seed
from torch.cuda import manual_seed_all as torch_cuda_manual_seed_all
import torch.nn.functional as F


from pytorch.models import FeedforwardNeuralNetwork


LOSSES = {"mean_squared_error": F.mse_loss}
OPTIMIZERS = {"rmsprop": optim.RMSprop}

logger = logging.getLogger(__name__)


def train(
    train_dataloader: object,
    val_dataloader: object,
    n_features: int,
    hidden_dim: int = None,
    final_activation: str = "linear",
    loss_name: str = "mean_squared_error",
    nb_epoch: int = 250,
    learning_rate: float = 0.005,
    ffn_depth: int = 2,
    optim_name: str = "rmsprop",
    seed: int = 1,
) -> object:
    """performs the training of FeedforwardNeuralNetwork

    Args:
        train_dataloader : an instance of the torch DataLoader class containing
            the training data
        val_dataloader : an instance of the torch DataLoader class containing
            the validating data
        n_features : dimension of an example, i.e. the number of feature
        hidden_dim : hidden layer size. Defaults to None.
        final_activation : activation function's name of the last layer. The
                possible choices are listed in ACTIVATION_FUNCTION. Defaults to
                "linear".
        loss_name : loss' name. The possible choices are listed in LOSSES.
            Defaults to "mean_squared_error".
        nb_epoch : number of epochs. Defaults to 250.
        learning_rate : learning rate. Defaults to 0.005.
        ffn_depth : number of hidden layers. Defaults to 2.
        optim_name : optimizer's name. The possible choices are listed in
            OPTIMIZERS. Defaults to "rmsprop".
    Return:
        the trained model
    """
    set_seed(seed)

    loss_func = LOSSES[loss_name]
    model = FeedforwardNeuralNetwork(
        n_input=n_features, hidden_dim=hidden_dim, final_activation=final_activation,
    )
    opt = OPTIMIZERS[optim_name](model.parameters(), lr=learning_rate)
    _fit(nb_epoch, model, loss_func, opt, train_dataloader, val_dataloader)
    return model


def _fit(nb_epoch, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(nb_epoch):
        model.train()
        for xb, yb in train_dl:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        _evaluate(model, loss_func, train_dl, valid_dl, epoch)


def _evaluate(model, loss_func, train_dl, valid_dl, epoch):
    model.eval()
    with no_grad():
        val_losses, val_nums = zip(*[_loss_batch(model, loss_func, xb, yb) for xb, yb in valid_dl])
        train_losses, train_nums = zip(
            *[_loss_batch(model, loss_func, xb, yb) for xb, yb in train_dl]
        )
    train_loss = np.sum(np.multiply(train_losses, train_nums)) / np.sum(train_nums)
    val_loss = np.sum(np.multiply(val_losses, val_nums)) / np.sum(val_nums)
    logger.info(
        f"epoch n°{epoch} completed\n" f"train_loss: {train_loss}\n" f"val_loss: {val_loss}"
    )


def _loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    return loss.item(), len(xb)


def set_seed(seed: int):
    np.random.seed(seed)
    torch_manual_seed(seed)
    torch_cuda_manual_seed_all(seed)
