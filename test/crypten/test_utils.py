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


@pytest.mark.parametrize(
    "tensors",
    [
        # return tensor
        th.tensor([1, 2, 3, 4]),
        # return tensor1, tensor2, tensor3
        (th.tensor([1, 2, 4, 5]), th.tensor([1.0, 2.0, 3.0]), th.tensor([5, 6, 7, 8])),
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


def test_pack_typerror_crypten_model():
    """
    Testing if we throw an error when trying to unpack an encrypted model,
    we should be able to unpack models that are not encrypted
    """
    dummy_input = th.rand(1, 28 * 28)
    expected_crypten_model = crypten.nn.from_pytorch(ExampleNet(), dummy_input)
    expected_crypten_model.encrypted = True

    with pytest.raises(TypeError):
        packed = utils.pack_values(expected_crypten_model)


def test_unpack_typerror_crypten_model():
    dummy_input = th.rand(1, 28 * 28)
    expected_crypten_model = crypten.nn.from_pytorch(ExampleNet(), dummy_input)
    packed = utils.pack_values(expected_crypten_model)

    with pytest.raises(TypeError):
        utils.unpack_values(packed)


def test_pack_other():
    expected_value = utils.pack_values(42)
    assert 42 == utils.unpack_values(expected_value)


def test_serialize_models():
    class ExampleNet(th.nn.Module):
        def __init__(self):
            super(ExampleNet, self).__init__()
            self.fc1 = th.nn.Linear(1024, 100)
            self.fc2 = th.nn.Linear(
                100, 2
            )  # For binary classification, final layer needs only 2 outputs

        def forward(self, x):
            out = self.fc1(x)
            out = th.nn.functional.relu(out)
            out = self.fc2(out)
            return out

    dummy_input = th.ones(1, 1024)
    example_net = ExampleNet()

    expected_output = example_net(dummy_input)

    onnx_bytes = utils.pytorch_to_onnx(example_net, dummy_input)
    crypten_model = utils.onnx_to_crypten(onnx_bytes)
    output = crypten_model(dummy_input)

    assert th.allclose(expected_output, output)
