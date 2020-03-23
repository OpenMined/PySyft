import crypten
import torch
from crypten.mpc import MPCTensor
from crypten.mpc.primitives import ArithmeticSharedTensor, BinarySharedTensor
from crypten.mpc.ptype import ptype as Ptype

from syft.frameworks.torch.tensors.crypten.syft_crypten import SyftCrypTensor

crypten.init()


def test_repr():
    x = torch.tensor([0.1, 0.2, 0.3])
    arithmetic = MPCTensor(x, ptype=Ptype.arithmetic)
    binary = MPCTensor(x, ptype=Ptype.binary)
    arithmetic_syft_crypten = SyftCrypTensor(tensor=arithmetic)
    binary_syft_crypten = SyftCrypTensor(tensor=binary)
    # Make sure these don't crash
    print(arithmetic_syft_crypten)
    print(arithmetic_syft_crypten.__repr__())
    print(binary_syft_crypten)
    print(binary_syft_crypten.__repr__())


def test_type():
    x = torch.tensor([0.1, 0.2, 0.3])
    ptype_values = [Ptype.arithmetic, Ptype.binary]
    tensor_types = [ArithmeticSharedTensor, BinarySharedTensor]
    for i, curr_ptype in enumerate(ptype_values):
        tensor = MPCTensor(x, ptype=curr_ptype)
        syft_crypten = SyftCrypTensor(tensor=tensor)
        assert isinstance(syft_crypten.tensor._tensor, tensor_types[i])


def test_to():
    # TODO need to check the internal structute if it is correctly changed or not!
    x = torch.tensor([0.1, 0.2, 0.3])
    ptype_values = [Ptype.arithmetic, Ptype.binary]
    tensor_types = [ArithmeticSharedTensor, BinarySharedTensor]
    for i, curr_ptype in enumerate(ptype_values):
        mpc = MPCTensor(x)
        tensor = SyftCrypTensor(tensor=mpc)
        syft_crypten1 = tensor.to(ptype_values[i])
        assert isinstance(syft_crypten1.tensor._tensor, tensor_types[i])


def test_binary():
    x = torch.tensor([0.1, 0.2, 0.3])
    tensor = MPCTensor(x)
    arithmetic_tensor = SyftCrypTensor(tensor=tensor)
    binary_tensor = arithmetic_tensor.binary()
    assert isinstance(binary_tensor.tensor._tensor, BinarySharedTensor)


def test_arithmetic():
    x = torch.tensor([0.1, 0.2, 0.3])
    tensor = MPCTensor(x, ptype=Ptype.binary)
    arithmetic_tensor = SyftCrypTensor(tensor=tensor)
    binary_tensor = arithmetic_tensor.arithmetic()
    assert isinstance(binary_tensor.tensor._tensor, ArithmeticSharedTensor)
