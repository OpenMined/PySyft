from ..abstract.worker import Worker


class Domain(Worker):
    def __init__(self, name):
        super().__init__(name=name)

    def _register_services(self) -> None:
        services = list()

        for s in services:
            self.msg_router[s.message_handler_type()] = s()
