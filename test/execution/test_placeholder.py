import syft as sy
import torch


def test_placeholder_expected_shape():
    @sy.func2plan(args_shape=[(3, 3), (3, 3)])
    def test_plan(x, y):
        return x + y

    for placeholder in test_plan.role.input_placeholders():
        assert placeholder.expected_shape == (3, 3)
