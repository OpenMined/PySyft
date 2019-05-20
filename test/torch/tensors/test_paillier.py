import pytest
import torch as th
from syft.frameworks.torch.tensors.interpreters import PaillierTensor
from syft.frameworks.torch.utils.paillier import PaillierEncryption
import syft as sy

def test_hook():
    assert sy.TorchHook(th)

def test_wrap():
    p = PaillierEncryption(length=14)
    x_tensor = th.Tensor([1, 2, 3])
    x = PaillierTensor(p.public_key).on(x_tensor)
    assert isinstance(x, th.Tensor)
    assert isinstance(x.child, PaillierTensor)
    assert isinstance(x.child.child, th.Tensor)

def test_encryption():
    p = PaillierEncryption(length=14)
    x_ = th.LongTensor([1,2,3])
    y_ = th.LongTensor([2,-2,3])
    x = PaillierTensor(p.public_key).on(x_)
    y = PaillierTensor(p.public_key).on(y_)
    x.child.encryptTensor(p.getEncryptedTensors())
    y.child.encryptTensor(p.getEncryptedTensors())
    assert x.child.encrypted
    assert x.child.encrypted


def test_encryption_decryption():
    p = PaillierEncryption(length=14)
    x_ = th.LongTensor([1,-2,3])
    x = PaillierTensor(p.public_key).on(x_)
    x.child.encryptTensor(p.getEncryptedTensors())
    to_be_asserted = PaillierTensor(p.public_key).on(th.LongTensor([1,-2,3])).child.child
    decrypted = x.child.decryptTensor(p.private_key, p.getEncryptedTensors()).child
    assert (to_be_asserted == decrypted).all()


def test_add_method():
    p = PaillierEncryption(length=14)
    x_ = th.LongTensor([1,2,3])
    y_ = th.LongTensor([2,-2,3])
    x = PaillierTensor(p.public_key).on(x_)
    y = PaillierTensor(p.public_key).on(y_)
    x.child.encryptTensor(p.getEncryptedTensors())
    y.child.encryptTensor(p.getEncryptedTensors())
    z = x.child.add(y,p.getEncryptedTensors())
    decrypted = z.child.decryptTensor(p.private_key, p.getEncryptedTensors()).child
    assert (decrypted == th.LongTensor([3, 0, 6])).all()


def test_sub_method():
    p = PaillierEncryption(length=14)
    x_ = th.LongTensor([1,2,3])
    y_ = th.LongTensor([2,-2,3])
    x = PaillierTensor(p.public_key).on(x_)
    y = PaillierTensor(p.public_key).on(y_)
    x.child.encryptTensor(p.getEncryptedTensors())
    y.child.encryptTensor(p.getEncryptedTensors())
    z = x.child.sub(y,p.getEncryptedTensors())
    decrypted = z.child.decryptTensor(p.private_key, p.getEncryptedTensors()).child
    assert (decrypted == th.LongTensor([-1,  4,  0])).all()


def test_mul_method():
    p = PaillierEncryption(length=14)
    x_ = th.LongTensor([2,-2,3])
    x = PaillierTensor(p.public_key).on(x_)
    x.child.encryptTensor(p.getEncryptedTensors())
    res = x.child.mul(4,p.getEncryptedTensors())
    z = th.LongTensor([[8, -8, 12]])
    decrypted = res.child.decryptTensor(p.private_key, p.getEncryptedTensors()).child
    assert (decrypted == z).all()

def test_div_method():
    p = PaillierEncryption(length=14)
    x_ = th.LongTensor([2,-2,3])
    x = PaillierTensor(p.public_key).on(x_)
    x.child.encryptTensor(p.getEncryptedTensors())
    res = x.child.div(0.2,p.getEncryptedTensors())
    z = th.LongTensor([[10, -10, 15]])
    decrypted = res.child.decryptTensor(p.private_key, p.getEncryptedTensors()).child
    assert (decrypted== z).all()
