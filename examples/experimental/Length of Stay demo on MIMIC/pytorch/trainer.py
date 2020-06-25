"""
Module containing functions useful for training a pytorch neural network
"""
import numpy as np
import logging
import torch
import torch.optim as optim
import torch.nn.functional as F


from pytorch.models import FeedforwardNeuralNetwork

# The loss functions used must be averaged over the batch to be consistent with
# the calculation of the metrics.
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
    for epoch in range(nb_epoch):
        model.train()
        for xb, yb in train_dataloader:
            loss = loss_func(model(xb), yb)
            loss.backward()
            opt.step()
            opt.zero_grad()

        logger.info(f"Epoch n°{epoch} completed")
        test(model, loss_func, val_dataloader, loss_name="Validation")
        test(model, loss_func, train_dataloader, loss_name="Train")

    return model


def test(model, loss_func, dataloader, loss_name: str = None):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in dataloader:
            # data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_func(output, target, reduction="sum").item()  # sum up batch loss

    test_loss /= len(dataloader.dataset)

    logger.info(f"{loss_name} loss: {test_loss}")


def _loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)
    return loss.item(), len(xb)


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
