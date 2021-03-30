# stdlib
import operator
from typing import Callable
from typing import List
from typing import Tuple

# third party
import pytest
import torch

# syft absolute
import syft as sy


@pytest.fixture()
def inputs() -> Tuple[int, float, bool, torch.Tensor]:
    return (1, 1.5, True, torch.Tensor([1, 2, 3]))


@pytest.fixture()
def input_pointer_types() -> Tuple[str, str, str, str]:
    return ("IntPointer", "FloatPointer", "BoolPointer", "TensorPointer")


@pytest.fixture()
def equality_functions() -> List[Callable]:
    return [operator.eq, operator.eq, operator.eq, torch.equal]


def test_resolve_any_pointer_type(
    root_client: sy.VirtualMachineClient,
    inputs: Tuple[int, float, bool, torch.Tensor],
    input_pointer_types: Tuple[str, str, str, str],
    equality_functions: List[Callable],
) -> None:
    tuple_ptr = root_client.syft.lib.python.Tuple(inputs)

    for idx, elem in enumerate(inputs):
        remote_pointer = tuple_ptr[idx]

        assert type(remote_pointer).__name__ == "AnyPointer"
        resolved_pointer = remote_pointer.resolve_pointer_type()
        assert remote_pointer.id_at_location == resolved_pointer.id_at_location
        assert type(resolved_pointer).__name__ == input_pointer_types[idx]
        assert equality_functions[idx](resolved_pointer.get(), elem)


def test_resolve_union_pointer_type(
    root_client: sy.VirtualMachineClient,
    inputs: Tuple[int, float, bool, torch.Tensor],
    input_pointer_types: Tuple[str, str, str, str],
    equality_functions: List[Callable],
) -> None:
    list_ptr = root_client.syft.lib.python.List(list(inputs))

    for idx, remote_pointer in enumerate(list_ptr):
        assert (
            type(remote_pointer).__name__ == "FloatIntStringTensorParameterUnionPointer"
        )
        resolved_pointer = remote_pointer.resolve_pointer_type()
        assert remote_pointer.id_at_location == resolved_pointer.id_at_location
        assert type(resolved_pointer).__name__ == input_pointer_types[idx]
        assert equality_functions[idx](resolved_pointer.get(), inputs[idx])
