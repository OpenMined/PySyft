from ..node.common.action.garbage_collect_object_action import (
    GarbageCollectObjectAction,
)
from ..node.common.action.get_object_action import GetObjectAction
from ..common.uid import UID


class Pointer:
    def __init__(self, location, id_at_location=None):
        if id_at_location is None:
            id_at_location = UID()

        self.location = location
        self.id_at_location = id_at_location

    def get(self):
        obj_msg = GetObjectAction(
            obj_id=self.id_at_location, address=self.location, reply_to=None
        )
        return self.location.send_immediate_msg_with_reply(msg=obj_msg).obj

    def __del__(self):
        print("Deleted:" + str(self))
        obj_msg = GarbageCollectObjectAction(
            obj_id=self.id_at_location, address=self.location
        )

        self.location.send_eventual_msg_without_reply(msg=obj_msg)
