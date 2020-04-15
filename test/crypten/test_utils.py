import pytest
import torch as th
import crypten
from syft.frameworks.crypten import utils
from syft.frameworks import crypten as syft_crypten


@pytest.mark.parametrize(
    "tensors",
    [
        # return tensor
        th.tensor([1, 2, 3, 4]),
        # return tensor1, tensor2, tensor3
        (th.tensor([1, 2, 4, 5]), th.tensor([1.0, 2.0, 3.0,]), th.tensor([5, 6, 7, 8])),
    ],
)
def test_pack_tensors(tensors):
    packed = utils.pack_values(tensors)
    unpacked = utils.unpack_values(packed)

    if isinstance(unpacked, tuple):  # return tensor1, tensor2 ...
        assert len(tensors) == len(unpacked)
        for unpacked_tensor, tensor in zip(unpacked, tensors):
            assert th.all(unpacked_tensor == tensor)

    else:  # return tensor
        assert th.all(unpacked == tensors)


def test_pack_crypten_model():
    class ExampleNet(th.nn.Module):
        def __init__(self):
            super(ExampleNet, self).__init__()
            self.fc = th.nn.Linear(28 * 28, 2)

        def forward(self, x):
            out = self.fc(x)
            return out

    dummy_input = th.rand(1, 28 * 28)
    expected_crypten_model = crypten.nn.from_pytorch(ExampleNet(), dummy_input)
    expected_out = expected_crypten_model(dummy_input)

    packed = utils.pack_values(expected_crypten_model)

    # zero all model's parameters
    with th.no_grad():
        for p in expected_crypten_model.parameters():
            assert isinstance(p, th.Tensor)
            p.set_(th.zeros_like(p))

    crypten_model = utils.unpack_values(packed, model=expected_crypten_model)
    assert isinstance(crypten_model, syft_crypten._WrappedCryptenModel)

    out = crypten_model(dummy_input)
    assert th.all(expected_out == out)
