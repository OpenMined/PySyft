# third party
import pytest
import torch as th

# syft absolute
import syft as sy
from syft.core.common.uid import UID
from syft.core.node.common.action.function_or_constructor_action import (
    RunFunctionOrConstructorAction,
)

# TODO test execution
# TODO test permissions


def test_constructor_in_default_permissions() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    torch = alice_client.torch
    ptr = torch.cuda.is_available()

    # allow the permssion to get the pointer
    def get_permission(obj: object) -> None:
        key = alice_client.verify_key
        ro = alice.store[obj.id_at_location]  # type: ignore
        ro.read_permissions[key] = obj.id_at_location  # type: ignore

    ptr = torch.cuda.is_available()
    get_permission(ptr)

    assert ptr.get() == th.cuda.is_available()


def test_constructor_not_in_default_permissions() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    torch = alice_client.torch
    ptr = torch.random.initial_seed()

    with pytest.raises(Exception) as e:
        ptr.get()

    assert (
        str(e.value)
        == "You do not have permission to .get() this tensor. Please submit a request."
    )


def test_run_function_or_constructor_action_serde() -> None:
    alice = sy.VirtualMachine(name="alice")
    alice_client = alice.get_client()

    args = (
        th.tensor([1, 2, 3]).send(alice_client),
        th.tensor([4, 5, 5]).send(alice_client),
    )

    msg = RunFunctionOrConstructorAction(
        path="torch.Tensor.add",
        args=args,
        kwargs={},
        id_at_location=UID(),
        address=alice_client.address,
        msg_id=UID(),
    )

    blob = msg.serialize()

    msg2 = sy.deserialize(blob=blob)

    assert msg2.path == msg.path
    # FIXME this cannot be checked before we fix the Pointer serde problem (see _proto2object in Pointer)
    # assert msg2.args == msg.args
    assert msg2.kwargs == msg.kwargs
    assert msg2.address == msg.address
    assert msg2.id == msg.id
    assert msg2.id_at_location == msg.id_at_location
