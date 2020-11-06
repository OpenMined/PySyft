# third party
import pytest

# syft absolute
from syft.util import get_fully_qualified_name
from syft.util import get_subclasses
from syft.util import index_syft_by_module_name


def test_get_fully_qualified_name_exception() -> None:
    class Bad:
        def __class__(self) -> None:  # type: ignore
            return None

    with pytest.raises(Exception):
        get_fully_qualified_name(Bad())


def test_get_subclasses() -> None:
    class A:
        pass

    class B(A):
        pass

    subclasses = get_subclasses(obj_type=A)
    assert subclasses == [B]


def test_index_syft_by_module_name() -> None:
    Int = index_syft_by_module_name(fully_qualified_name="syft.lib.python.Int")
    assert Int(1) == 1
