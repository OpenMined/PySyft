# syft absolute
import syft as sy


def test_list() -> None:
    alice = sy.Domain(name="alice").get_root_client()
    list_obj = sy.lib.python.List([1])
    list_remote = list_obj.send(alice)
    list_remote.append(2)
    assert list_remote.get() == [1, 2]  # inplace ops
