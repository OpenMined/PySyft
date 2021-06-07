# third party
import torch

# syft absolute
import syft as sy


def test_protobuf_torch_size_serializer_deserializer() -> None:
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
