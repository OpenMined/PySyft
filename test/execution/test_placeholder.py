import syft as sy
import torch
import pytest

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


def test_placeholder_forwarding():
    class TestClass(object):
        def child_only(self):
            return "Method 1"

        def copy(self):
            return "Method 2"  # pragma: no cover

    placeholder = PlaceHolder()
    placeholder.instantiate(TestClass())

    # Should be forwarded to the child
    assert placeholder.child_only() == "Method 1"

    # Should be found in placeholder -- should not be forwarded
    assert placeholder.copy() != "Method 2"

    # Not found in placeholder or child
    with pytest.raises(AttributeError):
        placeholder.dummy_method()
