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


def test_encrypted_encrypted_add():
    """
    Test addition of two encrypted values
    """

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y = (x + x).decrypt(pri)

    assert ((x_tensor + x_tensor) == y).all()


def test_encrypted_decrypted_add():
    """
    Test addition of an encryptd and decrypted value
    """

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y = (x + x_tensor).decrypt(pri)

    assert ((x_tensor + x_tensor) == y).all()


def test_decrypted_encrypted_add():
    """
    Test the addition of a decrypted and encrypted value
    """

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y = (x_tensor + x).decrypt(pri)

    assert ((x_tensor + x_tensor) == y).all()


def test_encrypted_encrypted_sub():

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y_tensor = torch.Tensor([2, 2, 2])
    y = y_tensor.encrypt(pub)

    z = (x - y).decrypt(pri)

    assert ((x_tensor - y_tensor) == z).all()


def test_encrypted_decrypted_sub():

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y_tensor = torch.Tensor([2, 2, 2])
    y = y_tensor.encrypt(pub)

    z = (x - y_tensor).decrypt(pri)

    assert ((x_tensor - y_tensor) == z).all()


def test_decrypted_encrypted_sub():

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y_tensor = torch.Tensor([2, 2, 2])
    y = y_tensor.encrypt(pub)

    z = (x_tensor - y).decrypt(pri)

    assert ((x_tensor - y_tensor) == z).all()


def test_encrypted_decrypted_mul():

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y_tensor = torch.Tensor([2, 2, 2])
    y = y_tensor.encrypt(pub)

    z = (x * y_tensor).decrypt(pri)

    assert ((x_tensor * y_tensor) == z).all()


def test_decrypted_encrypted_mul():

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y_tensor = torch.Tensor([2, 2, 2])
    y = y_tensor.encrypt(pub)

    z = (x_tensor * y).decrypt(pri)

    assert ((x_tensor * y_tensor) == z).all()


def test_encrypted_decrypted_matmul():

    pub, pri = sy.keygen()

    x_tensor = torch.tensor([1, 2, 3])
    x = x_tensor.encrypt(pub)

    y_tensor = torch.tensor([2, 2, 2])
    y = y_tensor.encrypt(pub)

    z = (x.mm(y_tensor)).decrypt(pri)

    assert (z == 12).all()


def test_decrypted_encrypted_matmul():

    pub, pri = sy.keygen()

    x_tensor = torch.Tensor([[1, 2, 3]])
    x = x_tensor.encrypt(pub)

    y_tensor = torch.Tensor([[2], [2], [2]])
    y = y_tensor.encrypt(pub)

    z = (x_tensor.mm(y)).decrypt(pri)

    assert ((x_tensor.mm(y_tensor)) == z).all()
