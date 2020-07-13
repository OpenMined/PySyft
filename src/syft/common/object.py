from .id import UID


class ObjectWithId(object):
    def __init__(self):
        self.id = UID()
