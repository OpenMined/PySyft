# stdlib
from collections import Counter
from collections import defaultdict
from typing import Any
from typing import Optional

# relative
from ....core.common.serde.serializable import serializable


@serializable(recursive_serde=True)
class DefaultDict(defaultdict):
    """Aim: build a class that inherits all the methods from collections.defaultdict but is serializable"""

    __attr_allowlist__ = []  # type: ignore

    def __init__(self, factory: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(default_factory=factory, **kwargs)


@serializable(recursive_serde=True)
class SerializableCounter(Counter):

    __attr_allowlist__ = []  # type: ignore

    def __init__(self, iterable: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__(iterable, **kwargs)
