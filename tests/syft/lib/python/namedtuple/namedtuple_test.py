# third party
import torch

# syft absolute
from syft import serialize
from syft.core.common.serde.deserialize import _deserialize


def test_torch_return_types_serde() -> None:
    x = torch.Tensor([[1, 2], [1, 2]])
    y = x.mode()

    ser = serialize(y)
    de = _deserialize(blob=ser)

    assert (de.values == y.values).all()
    assert (de.indices == y.indices).all()


def test_torch_qr_serde() -> None:
    x = torch.Tensor([[1, 2], [1, 2]])
    y = x.qr()

    ser = serialize(y)
    de = _deserialize(blob=ser)

    assert (de.Q == y.Q).all()
    assert (de.R == y.R).all()
