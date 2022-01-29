# syft absolute
from syft.lib.util import full_name_with_name
from syft.lib.util import full_name_with_qualname


class TSTClass:
    pass


def test_full_name_with_name():
    assert full_name_with_name(TSTClass) == f"{TSTClass.__module__}.{TSTClass.__name__}"


def test_full_name_with_qualname():
    class LocalTSTClass:
        pass

    assert (
        full_name_with_qualname(LocalTSTClass)
        == f"{LocalTSTClass.__module__}.{LocalTSTClass.__qualname__}"
    )
