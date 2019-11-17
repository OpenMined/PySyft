import pytest
import torch
import syft as sy


def test_encrypt_and_decrypt():
    """
    Test the basic paillier encrypt/decrypt functionality
    """

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y = x.decrypt(pri)

    assert (x_tensor == y).all()

def test_encrypted_add():
    """
    Test the basic paillier encrypt/decrypt functionality
    """

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y = (x + x).decrypt(pri)

    assert ((x_tensor+x_tensor) == y).all()