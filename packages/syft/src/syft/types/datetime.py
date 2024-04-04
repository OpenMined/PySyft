# stdlib
from datetime import datetime
from functools import total_ordering
from typing import Any

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_2
from .syft_object import SyftObject
from .uid import UID


@serializable()
@total_ordering
class DateTime(SyftObject):
    __canonical_name__ = "DateTime"
    __version__ = SYFT_OBJECT_VERSION_2

    id: UID | None = None  # type: ignore
    utc_timestamp: float

    @classmethod
    def now(cls) -> Self:
        return cls(utc_timestamp=datetime.utcnow().timestamp())

    def __str__(self) -> str:
        utc_datetime = datetime.utcfromtimestamp(self.utc_timestamp)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S")

    def __hash__(self) -> int:
        return hash(self.utc_timestamp)

    def __sub__(self, other: "DateTime") -> "DateTime":
        res = self.utc_timestamp - other.utc_timestamp
        return DateTime(utc_timestamp=res)

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return False
        return self.utc_timestamp == other.utc_timestamp

    def __lt__(self, other: Self) -> bool:
        return self.utc_timestamp < other.utc_timestamp
