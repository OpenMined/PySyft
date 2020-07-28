import random

from ..nodes.common.action.get_object_action import GetObjectAction


class Pointer:
    def __init__(self, location, id_at_location=None):
        if id_at_location is None:
            id_at_location = random.randint(0, 1000)

        self.location = location
        self.id_at_location = id_at_location

    def get(self):
        obj_msg = GetObjectAction(
            obj_id=self.id_at_location, address=self.location.address, reply_to=None
        )
        return self.location.send_msg_with_reply(msg=obj_msg).obj
