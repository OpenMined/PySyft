import torch

import pyfe
from pyfe.context import Context
from pyfe.key_generator import KeyGenerator


def test_encryption():
    """
    Test .encrypt(protocol="fe") on torch tensor
    """

    x = torch.tensor([1, 2, 3])

    context = Context()
    key_generator = KeyGenerator(context)

    pk, msk = key_generator.setup(len(x.flatten().tolist()))  # This is very very bad

    x_enc = x.encrypt(protocol="fe", context=context, public_key=pk)

    assert isinstance(x_enc.child.child, pyfe.encrypted_vector.EncryptedVector)
