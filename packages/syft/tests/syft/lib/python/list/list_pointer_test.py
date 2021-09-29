# syft absolute
import syft as sy


def test_list(
    root_client: sy.VirtualMachineClient,
) -> None:
    list_obj = sy.lib.python.List([1])
    list_remote = list_obj.send(root_client)
    list_remote.append(2)
    assert list_remote.get() == [1, 2]  # inplace ops
