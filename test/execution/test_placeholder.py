import pytest
import torch as th

from syft.execution.placeholder import PlaceHolder


def test_placeholder_forwarding():
    class TestClass(object):
        def child_only(self):
            return "Method 1"

        def copy(self):
            return "Method 2" # noqa: F821

    placeholder = PlaceHolder()
    placeholder.instantiate(TestClass())

    # Should be forwarded to the child
    assert placeholder.child_only() == "Method 1"

    # Should be found in placeholder -- should not be forwarded
    assert placeholder.copy() != "Method 2"

    # Not found in placeholder or child
    with pytest.raises(AttributeError):
        placeholder.dummy_method()
