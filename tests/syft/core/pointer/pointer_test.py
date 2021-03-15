from io import StringIO
import sys

# third party
import pytest
import torch as th

# syft absolute
import syft as sy


def validate_output(data, data_ptr):
    old_stdout = sys.stdout
    sys.stdout = newstdout = StringIO()

    data_ptr.print()

    sys.stdout = old_stdout
    assert newstdout.getvalue().strip("\n") == str(repr(data))


@pytest.mark.slow
@pytest.mark.parametrize("with_verify_key", [True, False])
def test_make_pointable(with_verify_key: bool) -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()
    client = bob.get_client()

    ten = th.tensor([1, 2])
    ptr = ten.send(root_client, pointable=False)

    assert len(client.store) == 0

    if with_verify_key:
        ptr.update_searchability(target_verify_key=client.verify_key)
    else:
        ptr.update_searchability()

    assert len(client.store) == 1


@pytest.mark.slow
@pytest.mark.parametrize("with_verify_key", [True, False])
def test_make_unpointable(with_verify_key: bool) -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()
    client = bob.get_client()

    ten = th.tensor([1, 2])
    ptr = ten.send(root_client, pointable=False)

    if with_verify_key:
        ptr.update_searchability(target_verify_key=client.verify_key)
    else:
        ptr.update_searchability()

    assert len(client.store) == 1

    if with_verify_key:
        ptr.update_searchability(pointable=False, target_verify_key=client.verify_key)
    else:
        ptr.update_searchability(pointable=False)

    assert len(client.store) == 0


@pytest.mark.slow
def test_pointable_property() -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()
    client = bob.get_client()

    ten = th.tensor([1, 2])
    ptr = ten.send(root_client, pointable=False)
    assert len(client.store) == 0

    ptr.pointable = False
    assert len(client.store) == 0

    ptr.pointable = True
    assert len(client.store) == 1

    ptr.pointable = True
    assert len(client.store) == 1

    ptr.pointable = False
    assert len(client.store) == 0


@pytest.mark.slow
@pytest.mark.xfail
def test_tags() -> None:
    bob = sy.VirtualMachine(name="Bob")
    root_client = bob.get_root_client()

    ten = th.tensor([1, 2])

    ten = ten.tag("tag1", "tag1", "other")
    assert ten.tags == ["tag1", "other"]

    # .send without `tags` passed in
    ptr = ten.send(root_client)
    assert ptr.tags == ["tag1", "other"]

    # .send with `tags` passed in
    ptr = ten.send(root_client, tags=["tag2", "tag2", "other"])
    assert ten.tags == ["tag2", "other"]
    assert ptr.tags == ["tag2", "other"]

    th.Tensor([1, 2, 3]).send(root_client, pointable=True, tags=["a"])
    th.Tensor([1, 2, 3]).send(root_client, pointable=True, tags=["b"])
    th.Tensor([1, 2, 3]).send(root_client, pointable=True, tags=["c"])
    th.Tensor([1, 2, 3]).send(root_client, pointable=True, tags=["d"])
    sy.lib.python.Int(2).send(root_client, pointable=True, tags=["e"])
    sy.lib.python.List([1, 2, 3]).send(root_client, pointable=True, tags=["f"])

    a = root_client.store["a"]
    b = root_client.store["b"]
    c = root_client.store["c"]
    d = root_client.store["d"]
    e = root_client.store["e"]

    result_ptr = a.requires_grad
    assert result_ptr.tags == ["a", "requires_grad"]

    result_ptr = b.pow(e)
    assert result_ptr.tags == ["b", "e", "pow"]

    result_ptr = c.pow(exponent=e)
    assert result_ptr.tags == ["c", "e", "pow"]

    result_ptr = root_client.torch.pow(d, e)
    assert result_ptr.tags == ["d", "e", "pow"]

    result_ptr = root_client.torch.pow(d, 3)
    assert result_ptr.tags == ["d", "pow"]

    # __len__ auto gets if you have permission
    f_root = root_client.store["f"]
    assert len(f_root) == 3


def test_issue_5170() -> None:
    bob = sy.VirtualMachine(name="Bob")
    client = bob.get_client()
    sy.lib.python.List([1, 2, 3]).send(client, pointable=True, tags=["f"])

    f_guest = client.store["f"]
    result_ptr = f_guest.len()
    assert result_ptr is not None
    assert result_ptr.tags == ["f", "__len__"]

    with pytest.raises(ValueError) as e:
        f_guest.__len__()

    assert str(e.value) == "Request to access data length rejected."


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


def test_printing() -> None:
    bob = sy.VirtualMachine(name="Bob")
    data_types = [
        sy.lib.python.Int(1),
        sy.lib.python.Float(1.5),
        sy.lib.python.Bool(True),
        sy.lib.python.List([1, 2, 3]),
        sy.lib.python.Tuple((1, 2, 3)),
        th.tensor([1, 2, 3]),
    ]

    root_client = bob.get_root_client()
    for data in data_types:
        validate_output(data, data.send(root_client))

    basic_client = bob.get_client()
    for data in data_types:
        validate_output(data, data.send(basic_client))


def test_printing_remote_creation() -> None:
    def create_data_types(client):
        return [
            client.syft.lib.python.Int(1),
            client.syft.lib.python.Float(1.5),
            client.syft.lib.python.Bool(True),
            client.syft.lib.python.List([1, 2, 3]),
            client.syft.lib.python.Tuple((1, 2, 3)),
            client.torch.Tensor([1, 2, 3]),
        ]

    bob = sy.VirtualMachine()
    results = []

    root_client = bob.get_root_client()
    for elem in create_data_types(root_client):
        out = elem.get(delete_obj=False)
        results.append(out)
        validate_output(out, elem)

    basic_client = bob.get_client()
    for idx, elem in enumerate(create_data_types(basic_client)):
        # shouldn't this fail?
        validate_output(results[idx], elem)
