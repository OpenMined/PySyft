import pytest
import torch as th
import crypten
from syft.frameworks.crypten import utils


class ExampleNet(th.nn.Module):
    def __init__(self):
        super(ExampleNet, self).__init__()
        self.fc = th.nn.Linear(28 * 28, 2)

    def forward(self, x):
        out = self.fc(x)
        return out


@pytest.mark.parametrize("value_type", ("single", "tuple"))
def test_pack_tensors(value_type):
    tensors_to_pack = {
        "single": th.tensor([1, 2, 3, 4]),
        "tuple": (th.tensor([1, 2, 4]), th.tensor([1.0, 2.0, 3.0,]), th.tensor([5, 6, 7])),
    }

    tensors = tensors_to_pack[value_type]

    packed = utils.pack_values(tensors)
    unpacked = utils.unpack_values(packed)

    if isinstance(unpacked, tuple):  # return tensor1, tensor2 ...
        assert len(tensors) == len(unpacked)
        for unpacked_tensor, tensor in zip(unpacked, tensors):
            assert th.all(unpacked_tensor == tensor)

    else:  # return tensor
        assert th.all(unpacked == tensors)


def test_pack_crypten_model():
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

    out = crypten_model(dummy_input)
    assert th.all(expected_out == out)


def test_pack_crypten_model_assert():
    dummy_input = th.rand(1, 28 * 28)
    expected_crypten_model = crypten.nn.from_pytorch(ExampleNet(), dummy_input)

    # Set as encrypted
    expected_crypten_model.encrypted = True

    with pytest.raises(TypeError):
        utils.pack_values(expected_crypten_model)


def test_unpack_crypten_model_assert():
    dummy_input = th.rand(1, 28 * 28)
    expected_crypten_model = crypten.nn.from_pytorch(ExampleNet(), dummy_input)

    packed = utils.pack_values(expected_crypten_model)

    with pytest.raises(TypeError):
        utils.unpack_values(packed)


@pytest.mark.parametrize("value_type", ("int", "string", "float", "dict", "tuple", "list"))
def test_pack_value_other(value_type):
    values_to_pack = {
        "int": 10,
        "string": "pysyft + crypten",
        "float": 10.42,
        "dict": {"test1": 10, "test2": 10.32},
        "tuple": (10, 42),
        "list": [10, 42],
    }

    value = values_to_pack[value_type]

    packed = utils._pack_value(value)
    assert packed == (utils.PACK_OTHER, value)

    unpacked = utils._unpack_value(packed)
    assert unpacked == value
