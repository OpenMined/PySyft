# third party
import torch

# syft absolute
from syft.core.common.serde.deserialize import _deserialize
from syft.lib.python import ValuesIndices


def test_torch_valuesindices_serde():
    x = torch.Tensor([1, 2, 3])
    y = x.cummax(0)
    values = y.values
    indices = y.indices

    ser = y.serialize()
    vi = ValuesIndices(values=values, indices=indices)
    de = _deserialize(blob=ser)

    assert (de.values == y.values).all()
    assert (de.indices == y.indices).all()
    assert (vi.values == de.values).all()
    assert (vi.indices == de.indices).all()
