import torch

import syft
from syft.serde import protobuf
from syft.serde.torch.serde import TORCH_STR_DTYPE

from test.serde.serde_helpers import *


# float16 and bfloat16 aren't supported on CPU
# quantized types can't be created by conversion with `to`
# complex types are not yet implemented
dtypes = ["uint8", "int8", "int16", "int32", "int64", "float32", "float64", "bool", "qint8"]


@pytest.mark.parametrize("str_dtype", dtypes)
def test_serde_roundtrip_protobuf(str_dtype, workers):
    """Checks that tensors passed through serialization-deserialization stay same"""

    def compare(roundtrip, original):
        assert type(roundtrip) == torch.Tensor
        assert roundtrip.type() == original.type()
        assert roundtrip.data.equal(original.data)
        return True

    serde_worker = syft.hook.local_worker
    original_framework = serde_worker.framework
    serde_worker.framework = None

    tensor = torch.rand([10, 100]) * 16
    tensor = tensor.to(TORCH_STR_DTYPE[str_dtype])

    protobuf_tensor = protobuf.serde._bufferize(serde_worker, tensor)
    roundtrip_tensor = protobuf.serde._unbufferize(serde_worker, protobuf_tensor)

    serde_worker.framework = original_framework

    assert compare(roundtrip_tensor, tensor) is True
