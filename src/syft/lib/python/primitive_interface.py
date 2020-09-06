# stdlib
from typing import Any

# syft relative
from ...core.common import UID


class PyPrimitive:
    def __init__(self, **kwargs: Any) -> None:
        self._id: UID
