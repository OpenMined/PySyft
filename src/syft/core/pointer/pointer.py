import random

from ..message import GetObjectMessage


class Pointer:
    def __init__(self, location, id_at_location=None):
        if id_at_location is None:
            id_at_location = random.randint(0, 1000)

        self.location = location
        self.id_at_location = id_at_location

    def get(self):
        obj_msg = GetObjectMessage(id=self.id_at_location, msg_destination=self.location)
        return self.location.send_msg(obj_msg)
