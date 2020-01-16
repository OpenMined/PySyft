import pytest
import torch
import syft as sy


def test_numpy_add():
    """
    Test the basic paillier encrypt/decrypt functionality
    """

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y = x.decrypt(pri)

    assert (x_tensor == y).all()


