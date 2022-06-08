# third party
import pytest
import torch as th

# syft absolute
import syft as sy


@pytest.fixture(scope="function")
def tensor() -> th.Tensor:
    t1 = th.tensor([[1.0, -1.0], [1.0, -1.0]])
    # scale, zero_point = 1e-4, 2
    # dtype = th.qint32
    # tensor = th.quantize_per_tensor(t1, scale, zero_point, dtype)
    return t1


@pytest.mark.parametrize("apache_arrow_backend", [True, False])
def test_protobuf_tensor_serializer_deserializer(
    apache_arrow_backend: bool, tensor: th.Tensor
) -> None:
    sy.flags.APACHE_ARROW_SERDE = apache_arrow_backend
    tensor2 = sy.lib.torch.tensor_util.tensor_serializer(tensor)
    tensor2_serial = sy.lib.torch.tensor_util.tensor_deserializer(tensor2)
    assert tensor2_serial.is_quantized is False
    assert tuple(tensor2_serial.shape) == tuple(tensor.shape)
