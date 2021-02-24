# third party
import torch

# syft absolute
from syft import serialize
from syft.core.common.serde.deserialize import _deserialize
from syft.lib.python import ValuesIndices


def test_torch_valuesindices_serde() -> None:
    x = torch.Tensor([[1, 2], [1, 2]])
    y = x.mode()
    values = y.values
    indices = y.indices

    ser = serialize(y)
    # horrible hack, we shouldnt be constructing these right now anyway
    params = [None] * 17
    params[0] = values
    params[1] = indices
    vi = ValuesIndices(*params)
    de = _deserialize(blob=ser)

    assert (de.values == y.values).all()
    assert (de.indices == y.indices).all()
    assert (vi.values == de.values).all()
    assert (vi.indices == de.indices).all()


def test_torch_qr_serde() -> None:
    x = torch.Tensor([[1, 2], [1, 2]])
    y = x.qr()
    values = y.Q
    indices = y.R

    ser = serialize(y)
    # horrible hack, we shouldnt be constructing these right now anyway
    params = [None] * 17
    params[8] = values
    params[9] = indices
    vi = ValuesIndices(*params)
    de = _deserialize(blob=ser)

    assert (de.Q == y.Q).all()
    assert (de.R == y.R).all()
    assert (vi.Q == de.Q).all()
    assert (vi.R == de.R).all()
