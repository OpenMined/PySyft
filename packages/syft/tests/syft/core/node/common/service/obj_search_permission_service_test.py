# third party
import torch as th

# syft absolute
import syft as sy
from syft.core.node.common.service.obj_search_permission_service import (
    ImmediateObjectSearchPermissionUpdateService,
)
from syft.core.node.common.service.obj_search_permission_service import (
    ObjectSearchPermissionUpdateMessage,
)


def test_object_search_permissons_update_message_serde() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    ptr = th.tensor([1, 2, 3]).send(bob_phone_client)

    msg = ObjectSearchPermissionUpdateMessage(
        add_instead_of_remove=True,
        target_verify_key=bob_phone_client.verify_key,
        target_object_id=ptr.id_at_location,
        address=bob_phone_client.address,
    )

    blob = sy.serialize(msg)
    msg2 = sy.deserialize(blob=blob)

    assert msg.id == msg2.id
    assert msg.address == msg2.address
    assert msg.add_instead_of_remove == msg2.add_instead_of_remove
    assert msg.target_verify_key == msg2.target_verify_key
    assert msg.target_object_id == msg2.target_object_id


def test_object_search_permissons_update_execute_add() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    ptr = th.tensor([1, 2, 3]).send(bob_phone_client)

    msg = ObjectSearchPermissionUpdateMessage(
        add_instead_of_remove=True,
        target_verify_key=bob_phone_client.verify_key,
        target_object_id=ptr.id_at_location,
        address=bob_phone_client.address,
    )

    ImmediateObjectSearchPermissionUpdateService.process(
        node=bob_phone, msg=msg, verify_key=bob_phone.verify_key
    )

    assert (
        bob_phone.store[ptr.id_at_location].search_permissions[
            bob_phone_client.verify_key
        ]
        == msg.id
    )


def test_object_search_permissons_update_execute_remove() -> None:
    bob_phone = sy.Device(name="Bob's iPhone")
    bob_phone_client = bob_phone.get_client()

    ptr = th.tensor([1, 2, 3]).send(bob_phone_client)

    msg = ObjectSearchPermissionUpdateMessage(
        add_instead_of_remove=False,
        target_verify_key=bob_phone_client.verify_key,
        target_object_id=ptr.id_at_location,
        address=bob_phone_client.address,
    )

    bob_phone.store[ptr.id_at_location].search_permissions[bob_phone.verify_key] = None

    ImmediateObjectSearchPermissionUpdateService.process(
        node=bob_phone, msg=msg, verify_key=bob_phone.verify_key
    )

    assert (
        bob_phone_client.verify_key
        not in bob_phone.store[ptr.id_at_location].search_permissions
    )
