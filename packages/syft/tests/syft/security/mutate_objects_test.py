# stdlib
from copy import copy

# third party
import torch as th

# syft absolute
import syft as sy

# from syft.core.node.common.action.run_class_method_action import RunClassMethodAction


def test_store_object_mutation(
    client: sy.VirtualMachineClient, root_client: sy.VirtualMachineClient
) -> None:
    """Check if the store object can be mutated by another user."""

    # root creates two tensors one visible one not
    x = th.tensor([1, 2, 3])
    x_ptr = x.send(root_client, pointable=True, tags=["visible"])

    y = th.tensor([3, 6, 9])
    y_ptr = y.send(root_client, pointable=False, tags=["invisible"])

    # guest gets a pointer to the visible one
    guest_x = client.store[x_ptr.id_at_location]
    guest_x.add_(guest_x)

    # guest constructs a pointer to the guessed hidden object
    guest_y = copy(guest_x)
    guest_y.id_at_location = sy.common.UID.from_string(y_ptr.id_at_location.no_dash)
    guest_y.add_(guest_y)

    # guest user should not be able to mutate objects that don't belong to them
    x_result = x_ptr.get(delete_obj=False)
    assert all(x_result == x) is True

    y_result = y_ptr.get(delete_obj=False)
    assert all(y_result == y) is True

    # guest creates object which gives it write permission
    g = th.tensor([1, 1, 1])
    g_ptr = g.send(client)

    # root uses guest object
    xg_ptr = x_ptr + g_ptr
    result_before = xg_ptr.get(delete_obj=False)

    # guest should not be able to mutate new destination
    # which means that write permissions should not flow as a union of execution
    guest_xg = copy(guest_x)
    guest_xg.id_at_location = sy.common.UID.from_string(xg_ptr.id_at_location.no_dash)
    guest_xg.add_(guest_xg)

    result_after = xg_ptr.get(delete_obj=False)
    assert (result_before == result_after).all()

    # but root can
    xg_ptr.add_(xg_ptr)

    result_after = xg_ptr.get(delete_obj=False)
    assert (result_before * 2 == result_after).all()

    # object owner should be able to mutate their own objects
    x_ptr.add_(x_ptr)

    new_result = x_ptr.get(delete_obj=False)
    assert all(new_result == (x + x)) is True

    # object owner should be able to mutate their own objects
    y_ptr.add_(y_ptr)

    new_result = y_ptr.get(delete_obj=False)
    assert all(new_result == (y + y)) is True


# TODO: Fix
# def test_store_overwrite_key(
#     client: sy.VirtualMachineClient, root_client: sy.VirtualMachineClient
# ) -> None:
#     """Check if someone can overwrite any address."""

#     # root creates two tensors one visible one not
#     x = th.tensor([1, 2, 3])
#     x_ptr = x.send(root_client, pointable=True, tags=["visible"])

#     y = th.tensor([3, 6, 9])
#     y_ptr = y.send(root_client, pointable=False, tags=["invisible"])

#     # guest gets a pointer to the visible one
#     guest_x = client.store[x_ptr.id_at_location]
#     guest_x.add_(guest_x)

#     # guest constructs a pointer to the guessed hidden object
#     guest_y = copy(guest_x)
#     target_uid = sy.common.UID.from_string(y_ptr.id_at_location.no_dash)
#     guest_y.id_at_location = target_uid
#     guest_y.add_(guest_y)

#     # guest should not be able to overwrite a destination with RunClassMethodAction
#     cmd = RunClassMethodAction(
#         path="torch.Tensor.add_",
#         _self=guest_y,
#         args=[guest_y],
#         kwargs={},
#         id_at_location=target_uid,  # normal destination changed
#         address=client.address,
#     )
#     client.send_immediate_msg_without_reply(msg=cmd)

#     # y should not have changed
#     y_result = y_ptr.get(delete_obj=False)
#     assert all(y_result == y) is True
