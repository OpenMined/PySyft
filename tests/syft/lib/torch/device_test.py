# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.lib.python import String


def test_device() -> None:
    device = th.device("cuda")
    assert device.type == "cuda"
    assert device.index is None


def test_device_init() -> None:
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_client()
    torch = client.torch

    type_str = String("cuda:0")
    str_pointer = type_str.send(client)

    device_pointer = torch.device(str_pointer)
    assert type(device_pointer).__name__ == "devicePointer"
    assert isinstance(device_pointer.id_at_location, UID)
