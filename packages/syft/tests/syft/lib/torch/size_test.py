# third party
import pytest
import torch

# syft absolute
import syft as sy


@pytest.mark.parametrize("apache_arrow_backend", [True, False])
def test_protobuf_torch_size_serializer_deserializer(
    apache_arrow_backend: bool,
) -> None:
    sy.flags.APACHE_ARROW_SERDE = apache_arrow_backend
    torch_size = torch.Size([4, 5, 2, 1])
    torch_size_pb = sy.lib.torch.size.protobuf_torch_size_serializer(torch_size)
    torch_size_deserialized = sy.lib.torch.size.protobuf_torch_size_deserializer(
        torch_size_pb
    )

    assert isinstance(torch_size_deserialized, torch.Size)
    assert torch_size == torch_size_deserialized


def test_torch_size_serde(client: sy.VirtualMachineClient) -> None:
    torch_size = torch.Size([2, 3, 4, 100])
    torch_size_ptr = torch_size.send(client)
    assert torch_size_ptr.get() == torch_size
