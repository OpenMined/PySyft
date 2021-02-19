# third party
import pytest
import torch as th

# syft absolute
import syft as sy


@pytest.fixture(scope="function")
def tensor() -> th.Tensor:
    t1 = th.tensor([[1.0, -1.0], [1.0, -1.0]])
    scale, zero_point = 1e-4, 2
    dtype = th.qint32
    tensor = th.quantize_per_tensor(t1, scale, zero_point, dtype)
    return tensor


def test_protobuf_tensor_serializer_deserializer(tensor: th.Tensor) -> None:
    tensor2 = sy.lib.torch.tensor_util.protobuf_tensor_serializer(tensor)
    assert tensor2.is_quantized is True
    assert tuple(tensor2.shape) == tuple(tensor.shape)
    tensor2_serial = sy.lib.torch.tensor_util.protobuf_tensor_deserializer(tensor2)
    assert tensor2_serial.is_quantized is True
    assert tuple(tensor2_serial.shape) == tuple(tensor.shape)
