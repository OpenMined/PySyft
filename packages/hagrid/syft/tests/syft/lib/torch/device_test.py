# stdlib
from typing import Any

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.lib.python import String


def test_device() -> None:
    device = th.device("cuda")
    assert device.type == "cuda"
    assert device.index is None


def test_device_init(node: sy.VirtualMachine, client: sy.VirtualMachineClient) -> None:
    assert node.name == "Bob"
    torch = client.torch

    type_str = String("cuda:0")
    str_pointer = type_str.send(client)

    device_pointer = torch.device(str_pointer)
    assert type(device_pointer).__name__ == "devicePointer"
    assert isinstance(device_pointer.id_at_location, UID)


@pytest.mark.slow
@pytest.mark.parametrize("type_str", ["cpu", "cuda"])
@pytest.mark.parametrize("index", [None, 0])
@pytest.mark.parametrize("apache_arrow_backend", [True, False])
def test_device_serde(
    apache_arrow_backend: bool,
    type_str: str,
    index: Any,
    root_client: sy.VirtualMachineClient,
) -> None:
    sy.flags.APACHE_ARROW_SERDE = apache_arrow_backend
    device = th.device(type_str, index)
    device_ptr = device.send(root_client)
    assert device_ptr.get() == device
