# stdlib
import time

# syft absolute
import syft as sy
from syft.core.node.common.action.save_object_action import SaveObjectAction
from syft.core.store.storeable_object import StorableObject


def test_unfinished_task(get_clients) -> None:
    client = get_clients(1)[0]

    list_pointer = sy.lib.python.List().send(client)
    int_pointer = sy.lib.python.Int(1).send(client)
    time.sleep(5)
    int_obj = int_pointer.get()
    list_pointer.append(int_pointer)
    storeable_object = StorableObject(id=int_pointer.id_at_location, data=int_obj)
    save_object_action = SaveObjectAction(obj=storeable_object, address=client.address)
    client.send_immediate_msg_without_reply(msg=save_object_action)
    time.sleep(5)
    assert list_pointer.get() == [1]
