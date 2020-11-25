# third party
import pytest
from sympc.tensor import FixedPrecisionTensor
import torch

# syft absolute
import syft as sy


def test_fpt_send() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    fpt = FixedPrecisionTensor(data=50, encoder_precision=4, encoder_base=10)

    ptr = fpt.send(alice_client)

    assert fpt == ptr.get()
