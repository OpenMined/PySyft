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
        ptr.update_searchability(target_verify_key=client.verify_key)
    else:
        ptr.update_searchability()

    assert len(client.store) == 1


@pytest.mark.parametrize("with_verify_key", [True, False])
def test_make_unsearchable(with_verify_key: bool) -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()
    client = bob.get_client()

    ten = th.tensor([1, 2])
    ptr = ten.send(root_client)

    if with_verify_key:
        ptr.update_searchability(target_verify_key=client.verify_key)
    else:
        ptr.update_searchability()

    assert len(client.store) == 1

    if with_verify_key:
        ptr.update_searchability(searchable=False, target_verify_key=client.verify_key)
    else:
        ptr.update_searchability(searchable=False)

    assert len(client.store) == 0


def test_searchable_property() -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()
    client = bob.get_client()

    ten = th.tensor([1, 2])
    ptr = ten.send(root_client)
    assert len(client.store) == 0

    ptr.searchable = False
    assert len(client.store) == 0

    ptr.searchable = True
    assert len(client.store) == 1

    ptr.searchable = True
    assert len(client.store) == 1

    ptr.searchable = False
    assert len(client.store) == 0


def test_tags() -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()

    ten = th.tensor([1, 2])

    ten = ten.tag("tag1")
    assert ten.tags == ["tag1"]

    # .send without `tags` passed in
    ptr = ten.send(root_client)
    assert ptr.tags == ["tag1"]

    # .send with `tags` passed in
    ptr = ten.send(root_client, tags=["tag2"])
    assert ten.tags == ["tag2"]
    assert ptr.tags == ["tag2"]


def test_description() -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()

    ten = th.tensor([1, 2])

    ten = ten.describe("description 1")
    assert ten.description == "description 1"

    # .send without `description` passed in
    ptr = ten.send(root_client)
    assert ptr.description == "description 1"

    # .send with `description` passed in
    ptr = ten.send(root_client, description="description 2")
    assert ten.description == "description 2"
    assert ptr.description == "description 2"
