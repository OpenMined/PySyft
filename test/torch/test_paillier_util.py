from syft.frameworks.torch.utils.paillier import PaillierEncryption
import phe
import pytest
import torch
import syft
import random


def test_attributes():
    p = PaillierEncryption()
    assert hasattr(p, "public_key")
    assert hasattr(p, "private_key")
    assert hasattr(p, "id")
    assert hasattr(p, "owner")
    assert hasattr(p, "tensors_encrypted")

def test_public_key():
    p = PaillierEncryption()
    assert isinstance(p.public_key, phe.paillier.PaillierPublicKey)

def test_private_key():
    p = PaillierEncryption()
    assert isinstance(p.private_key, phe.paillier.PaillierPrivateKey)

def test_encryption_type():
    p = PaillierEncryption()
    encrypted_num = p.public_key.encrypt(123)
    assert isinstance(encrypted_num, phe.paillier.EncryptedNumber)

def test_positive_int_encryption_decryption():
    p = PaillierEncryption()
    num = int(1e4 * random.random())
    encrypted = p.public_key.encrypt(num)
    decrypted = p.private_key.decrypt(encrypted)
    assert decrypted == num

def test_negative_int_encryption_decryption():
    p = PaillierEncryption()
    num = int(1e4 * random.random()) * -1
    encrypted = p.public_key.encrypt(num)
    decrypted = p.private_key.decrypt(encrypted)
    assert decrypted == num
