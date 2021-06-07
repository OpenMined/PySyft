# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="zksk")
def test_secret_serde() -> None:
    vm = sy.VirtualMachine()
    client = vm.get_root_client()

    # third party
    import zksk as zk

    sy.load("zksk")

    r = zk.Secret(zk.utils.get_random_num(bits=128))
    r_ptr = r.send(client)
    r2 = r_ptr.get()

    assert r == r2
