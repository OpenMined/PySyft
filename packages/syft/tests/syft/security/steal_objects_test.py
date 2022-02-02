# third party
import pytest
import torch as th

# syft absolute
import syft as sy


def test_steal_data_through_mutation(
    client: sy.VirtualMachineClient,
    root_client: sy.VirtualMachineClient,
) -> None:
    secret = th.Tensor([1, 2, 3])
    secret.send(root_client, tags=["secret"])

    secret_ptr = client.store[0]
    with pytest.raises(Exception):
        secret_ptr.get()

    # 💰🔒🎩🚓
    # we can try to create a mutable object which we own
    list_obj = sy.lib.python.List([])
    list_obj_ptr = list_obj.send(client, tags=["swag"])
    list_obj_ptr.append(1)
    bag = list_obj_ptr.get(delete_obj=False)
    assert len(bag) == 1
    assert bag[0] == 1

    # and then mutate it with someone elses secret data
    # stdlib
    list_obj_ptr.append(secret_ptr)

    with pytest.raises(Exception):
        _ = list_obj_ptr.get()

    # before fix this was possible
    # see run_class_method_action.py: 235
    # assert len(swag) > 1
    # assert (swag[1] == secret).all()
