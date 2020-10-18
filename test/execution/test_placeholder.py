import syft as sy
import torch

from syft.execution.placeholder import PlaceHolder


def test_placeholder_expected_shape():
    @sy.func2plan(args_shape=[(3, 3), (3, 3)])
    def test_plan(x, y):
        return x + y

    for placeholder in test_plan.role.input_placeholders():
        assert placeholder.expected_shape == (3, 3)


def test_create_from():
    t = torch.tensor([1, 2, 3])
    ph = PlaceHolder.create_from(t)

    assert isinstance(ph, PlaceHolder)
    assert (ph.child == torch.tensor([1, 2, 3])).all()
