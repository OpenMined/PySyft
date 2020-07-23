from ..abstract.node import Node


class Network(Node):
    def __init__(self, name: str):
        super().__init__(name=name)

    def _register_services(self) -> None:
        services = list()

        for s in services:
            self.msg_router[s.message_handler_type()] = s()
