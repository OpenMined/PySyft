# third party
import pytest
import torch as th

# syft absolute
import syft as sy


@pytest.mark.parametrize("with_verify_key", [True, False])
def test_make_searchable(with_verify_key: bool) -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()
    client = bob.get_client()

    ten = th.tensor([1, 2])
    ptr = ten.send(root_client)

    assert len(client.store) == 0

    if with_verify_key:
        ptr.make_searchable(target_verify_key=client.verify_key)
    else:
        ptr.make_searchable()

    assert len(client.store) == 1


def test_searchable() -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()
    client = bob.get_client()

    ten = th.tensor([1, 2])
    _ = ten.send(root_client)

    assert len(client.store) == 0

    _ = ten.send(root_client, searchable=True)

    assert len(client.store) == 1
