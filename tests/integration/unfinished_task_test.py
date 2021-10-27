# syft absolute
from syft.core.node.common.action.save_object_action import SaveObjectAction
from syft.core.store.storeable_object import StorableObject


def test_tensor_abstraction_pointer(get_clients) -> None:
    client = get_clients(1)[0]

    list_pointer = client.syft.lib.python.List()
    int_pointer = client.syft.lib.python.Int(1)
    int_obj = int_pointer.get()
    list_pointer.append(int_pointer)
    storeable_object = StorableObject(id=int_pointer.id_at_location, data=int_obj)
    save_object_action = SaveObjectAction(
        obj=storeable_object, address=int_pointer.address
    )
    client.send_immediate_msg_without_reply(msg=save_object_action)

    assert list_pointer.get() == [1]
