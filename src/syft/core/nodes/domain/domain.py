from ..abstract.worker import Worker


class Domain(Worker):
    def __init__(self, name):
        super().__init__(name=name)

        available_device_types = set()
        # TODO: add available compute types

        default_device = None
        # TODO: add default compute type
