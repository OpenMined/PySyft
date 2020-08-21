"""In this test suite, we evaluate the operations done on a remote tensor
(we use the the pointer to a tensor).
For more info on the remote tensor please see the documentation syft/lib/torch
"""

from syft.core.pointer.pointer import Pointer
from syft.lib.torch.tensor_util import TORCH_STR_DTYPE

import syft as sy
import torch as th
import pytest
from itertools import product

# Currently, we do not have constructors with torch.Tensor
# for dtype in ["complex*", "q*"] (complex and quantized types)
TYPES_EXCEPTIONS_PREFIX = ("complex", "q")

TEST_TYPES = [
    e for e in TORCH_STR_DTYPE.keys() if not e.startswith(TYPES_EXCEPTIONS_PREFIX)
]
BASIC_OPS = ["__add__", "add", "__sub__", "sub", "__mul__", "mul"]

TEST_DATA = list(product(TEST_TYPES, BASIC_OPS))


def test_torch_remote_tensor_register() -> None:
    """ Test if sending a tensor will be registered on the remote worker. """

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    x = th.tensor([-1, 0, 1, 2, 3, 4])
    ptr = x.send(alice_client)

    assert len(alice.store) == 1

    ptr = x.send(alice_client)
    assert len(alice.store) == 1  # Same id

    ptr.get()
    assert len(alice.store) == 0  # Get removes the object


@pytest.mark.parametrize("tensor_type, op_name", TEST_DATA)
def test_torch_remote_remote_basic_ops(tensor_type: str, op_name: str) -> None:
    """ Test basic operations on remote simple tensors """

    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    t_type = TORCH_STR_DTYPE[tensor_type]
    x = th.tensor([-1, 0, 1, 2, 3, 4], dtype=t_type)

    xp = x.send(alice_client)

    op_method = getattr(xp, op_name, None)
    assert op_method

    result = op_method(xp)  # op(xp, xp)

    assert isinstance(result, Pointer)
    local_result = result.get()

    # TODO: put thought into garbage collection and then
    #  uncoment this.
    # del xp
    #
    # assert len(alice.store.) == 0


def test_torch_serde() -> None:

    x = th.tensor([1.0, 2, 3, 4], requires_grad=True)

    # This is not working currently:
    # (x * 2).sum().backward()
    # But pretend we have .grad
    x.grad = th.randn_like(x)

    blob = x.serialize()

    x2 = sy.deserialize(blob=blob)

    assert (x == x2).all()


def test_torch_no_read_permissions() -> None:

    bob = sy.VirtualMachine(name="bob")
    root_bob = bob.get_root_client()
    guest_bob = bob.get_client()

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_bob)

    # guest creates a pointer to that object (assuming the client can guess/inpher the ID)
    ptr.client = guest_bob

    # this should trigger an exception
    with pytest.raises(Exception) as exception:
        local_x = ptr.get()

    assert str(exception.value) == "You do not have permission to .get() this tensor. Please submit a request."

    x = th.tensor([1, 2, 3, 4])

    # root user of Bob's machine sends a tensor
    ptr = x.send(root_bob)

    # but if root bob asks for it it should be fine
    x2 = ptr.get()

    assert (x == x2).all()

    assert x.grad == x2.grad
