from abc import ABC
from abc import abstractmethod

from syft.generic.object_storage import ObjectStore


class AbstractMessageHandler(ABC):
    def __init__(self, object_store: ObjectStore):
        self.object_store = object_store
        self.routing_table = self.init_routing_table()

    @abstractmethod
    def init_routing_table(self):
        return {}

    def supports(self, msg):
        return type(msg) in self.routing_table.keys()

    def handle(self, msg):
        return self.routing_table[type(msg)](msg)
