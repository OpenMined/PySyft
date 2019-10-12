from abc import ABC, abstractmethod


class MapperNeutral(ABC):
    @abstractmethod
    def to_neutral(self, obj) -> object:
        pass

    def from_neutral(self, obj) -> object:
        pass


class SerDe(ABC):

    @abstractmethod
    def serialize(self, obj, **kwargs) -> object:
        pass

    @abstractmethod
    def deserialize(self, serialized, **kwargs) -> object:
        pass
