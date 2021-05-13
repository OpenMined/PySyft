# stdlib
from io import StringIO
import sys
from typing import Any
from typing import List

# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.node.common.client import AbstractNodeClient
from syft.core.pointer.pointer import Pointer


def validate_output(data: Any, data_ptr: Pointer) -> None:
    old_stdout = sys.stdout
    sys.stdout = newstdout = StringIO()

    data_ptr.print()

    sys.stdout = old_stdout
    assert newstdout.getvalue().strip("\n") == str(repr(data))


def validate_permission_error(data_ptr: Pointer) -> None:
    old_stdout = sys.stdout
    sys.stdout = newstdout = StringIO()

    data_ptr.print()

    sys.stdout = old_stdout

    assert newstdout.getvalue().startswith("No permission to print")


@pytest.mark.slow
@pytest.mark.parametrize("with_verify_key", [True, False])
def test_make_pointable(
    with_verify_key: bool,
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
    root_client: sy.VirtualMachineClient,
) -> None:
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
def test_make_unpointable(
    with_verify_key: bool,
    node: sy.VirtualMachine,
    client: sy.VirtualMachineClient,
    root_client: sy.VirtualMachineClient,
) -> None:
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
def test_pointable_property(
    client: sy.VirtualMachineClient, root_client: sy.VirtualMachineClient
) -> None:
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
def test_tags(root_client: sy.VirtualMachineClient) -> None:
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


def test_auto_approve_length_request(client: sy.VirtualMachineClient) -> None:
    remote_list = sy.lib.python.List([1, 2, 3]).send(client)

    result_len_ptr = remote_list.len()
    assert result_len_ptr is not None
    assert result_len_ptr.get() == 3

    remote_list = client.syft.lib.python.List([1, 2, 3])
    result_len_ptr = remote_list.len()
    assert result_len_ptr is not None
    assert result_len_ptr.get() == 3


def test_description(root_client: sy.VirtualMachineClient) -> None:
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


def test_printing(
    client: sy.VirtualMachineClient, root_client: sy.VirtualMachineClient
) -> None:
    data_types = [
        sy.lib.python.Int(1),
        sy.lib.python.Float(1.5),
        sy.lib.python.Bool(True),
        sy.lib.python.List([1, 2, 3]),
        sy.lib.python.Tuple((1, 2, 3)),
        th.tensor([1, 2, 3]),
    ]

    for data in data_types:
        validate_output(data, data.send(root_client))

    for data in data_types:
        validate_permission_error(data.send(client))


@pytest.mark.slow
def test_printing_remote_creation(
    client: sy.VirtualMachineClient, root_client: sy.VirtualMachineClient
) -> None:
    def create_data_types(client: AbstractNodeClient) -> List[Pointer]:
        return [
            client.syft.lib.python.Int(1),
            client.syft.lib.python.Float(1.5),
            client.syft.lib.python.Bool(True),
            client.syft.lib.python.List([1, 2, 3]),
            client.syft.lib.python.Tuple((1, 2, 3)),
            client.torch.Tensor([1, 2, 3]),
        ]

    for elem in create_data_types(root_client):
        out = elem.get(delete_obj=False)
        validate_output(out, elem)

    for idx, elem in enumerate(create_data_types(client)):
        validate_permission_error(elem)


def test_exhausted(root_client: sy.VirtualMachineClient) -> None:
    int_ptr = root_client.syft.lib.python.Int(0)
    int_ptr.get()  # ptr gets exhausted after this call

    with pytest.raises(ReferenceError) as e:
        int_ptr.get()

    assert str(e.value) == "Object has already been deleted. This pointer is exhausted"
