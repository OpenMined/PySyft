from ..common.serde.serializable import bind_protobuf
from ..common.serde.recursive import RecursiveSerde
from collections import defaultdict
from collections import Counter


@bind_protobuf
class DefaultDict(defaultdict, RecursiveSerde):
    """Aim: build a class that inherits all the methods from collections.defaultdict but is serializable"""

    __attr_allowlist__ = []

    def __init__(self, factory=None, **kwargs):
        super().__init__(default_factory=factory, **kwargs)


@bind_protobuf
class SerializableCounter(Counter, RecursiveSerde):

    __attr_allowlist__ = []

    def __init__(self, iterable=None, **kwargs):
        super().__init__(iterable, **kwargs)
