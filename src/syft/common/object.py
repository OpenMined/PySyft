from .id import UID


class ObjectWithId:
    def __init__(self):
        self.id: UID = UID()
