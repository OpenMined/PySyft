# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.lib import load_lib


def test_searchable_pointable(root_client: sy.VirtualMachineClient) -> None:
    with pytest.deprecated_call():
        x_ptr = th.Tensor([1, 2, 3]).send(root_client, searchable=True)

    assert x_ptr.pointable is True

    with pytest.deprecated_call():
        assert x_ptr.searchable is True

    with pytest.deprecated_call():
        x_ptr.searchable = False
        assert x_ptr.searchable is False


@pytest.mark.slow
def test_load_lib_deprecated() -> None:
    with pytest.deprecated_call():
        assert load_lib("tenseal") is None
