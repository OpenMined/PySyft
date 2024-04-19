# stdlib
from datetime import datetime, timedelta, timezone
from functools import total_ordering

from typing import Any

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_2
from .syft_object import SyftObject
from .uid import UID


def td_format(td_object: timedelta) -> str:
    seconds = int(td_object.total_seconds())
    if seconds == 0:
        return "0 seconds"

    periods = [
        ("year", 60 * 60 * 24 * 365),
        ("month", 60 * 60 * 24 * 30),
        ("day", 60 * 60 * 24),
        ("hour", 60 * 60),
        ("minute", 60),
        ("second", 1),
    ]

    strings = []
    for period_name, period_seconds in periods:
        if seconds >= period_seconds:
            period_value, seconds = divmod(seconds, period_seconds)
            has_s = "s" if period_value > 1 else ""
            strings.append(f"{period_value} {period_name}{has_s}")

    return ", ".join(strings)


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

    def as_datetime(self) -> datetime:
        return datetime.utcfromtimestamp(self.utc_timestamp)

    def timeago(self) -> str:
        delta = datetime.utcnow() - self.as_datetime()
        return td_format(delta) + " ago"

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
