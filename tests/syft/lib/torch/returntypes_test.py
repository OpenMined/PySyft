# stdlib
from typing import Any

# third party
from packaging import version
import pytest
import torch

# syft absolute
import syft as sy
from syft.lib.torch.return_types import types_fields

A = torch.tensor([[1.0, 1, 1], [2, 3, 4], [3, 5, 2], [4, 2, 5], [5, 4, 3]])
B = torch.tensor([[-10.0, -3], [12, 14], [14, 12], [16, 16], [18, 16]])
x = torch.Tensor([[1, 2], [1, 2]])
s = torch.tensor(
    [[-0.1000, 0.1000, 0.2000], [0.2000, 0.3000, 0.4000], [0.0000, -0.3000, 0.5000]]
)

torch_version_ge_1d5d0 = version.parse(
    torch.__version__.split("+")[0]
) >= version.parse("1.5.0")

parameters = [
    ("eig", x, True),
    ("kthvalue", x, 1),
    ("lstsq", A, B),
    ("slogdet", x, None),
    ("qr", x, None),
    ("mode", x, None),
    ("solve", s, s),
    ("sort", s, None),
    ("symeig", s, None),
    ("topk", s, 1),
    ("triangular_solve", s, s),
    ("svd", s, None),
    ("geqrf", s, None),
    ("median", s, 0),
    ("max", s, 0),
    ("min", s, 0),
]

if torch_version_ge_1d5d0:
    parameters.append(("cummax", x, 0))
    parameters.append(("cummin", x, 0))


def assert_eq(y1, y2):  # type: ignore
    fields = types_fields[type(y1)]
    for field in fields:
        assert (getattr(y1, field) == getattr(y2, field)).all()


def assert_serde(y):  # type: ignore
    ser = sy.serialize(y)
    de = sy.deserialize(ser)
    assert_eq(y, de)


@pytest.mark.parametrize("op_name, ten, _arg", parameters)
def test_returntypes(
    op_name: str,
    ten: torch.Tensor,
    _arg: Any,
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
) -> None:
    # serde y=ten.op(_arg)
    op = getattr(ten, op_name)
    if _arg is not None:
        y = op(_arg)
    else:
        y = op()

    assert_serde(y)  # type: ignore

    # serde y=torch.Tensor.op(ten, _arg)
    op = getattr(torch.Tensor, op_name)
    if _arg is not None:
        y = op(ten, _arg)
    else:
        y = op(ten)

    assert_serde(y)  # type: ignore

    # ptr.op(_arg).get()
    ptr = ten.send(client)
    op = getattr(ptr, op_name)
    if _arg is not None:
        y_ptr = op(_arg)
    else:
        y_ptr = op()
    y_get = y_ptr.get()
    assert_eq(y, y_get)  # type: ignore

    # alice_client.torch.Tensor.op(ptr, _arg).get()
    op = getattr(client.torch.Tensor, op_name)
    if _arg is not None:
        y_ptr = op(ptr, _arg)
    else:
        y_ptr = op(ptr)
    y_get = y_ptr.get()
    assert_eq(y, y_get)  # type: ignore

    # TODO: Loot at https://github.com/OpenMined/PySyft/issues/5249
    # serde y=torch.op(ten, _arg)
    # alice_client.torch.op(ptr, _arg).get()
