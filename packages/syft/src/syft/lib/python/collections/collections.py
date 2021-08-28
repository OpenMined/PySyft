# stdlib
from collections import Counter
from collections import defaultdict
from typing import Any
from typing import Optional

# relative
from ....core.common.serde.recursive import RecursiveSerde
from ....core.common.serde.serializable import bind_protobuf


@bind_protobuf
class DefaultDict(defaultdict, RecursiveSerde):
    """Aim: build a class that inherits all the methods from collections.defaultdict but is serializable"""

    __attr_allowlist__ = []  # type: ignore

    def __init__(self, factory: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(default_factory=factory, **kwargs)


@bind_protobuf
class SerializableCounter(Counter, RecursiveSerde):

    __attr_allowlist__ = []  # type: ignore

    def __init__(self, iterable: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(iterable, **kwargs)
