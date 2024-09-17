# stdlib
from datetime import datetime
from datetime import timedelta
from datetime import timezone
from functools import total_ordering
import re
from typing import Any

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject
from .uid import UID

DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
DATETIME_REGEX = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}"


def str_is_datetime(str_: str) -> bool:
    return bool(re.match(DATETIME_REGEX, str_))


@serializable()
@total_ordering
class DateTime(SyftObject):
    __canonical_name__ = "DateTime"
    __version__ = SYFT_OBJECT_VERSION_1

    id: UID | None = None  # type: ignore
    utc_timestamp: float

    @classmethod
    def now(cls) -> Self:
        utc_datetime = datetime.now(tz=timezone.utc)
        return cls(utc_timestamp=utc_datetime.timestamp())

    @classmethod
    def from_str(cls, datetime_str: str) -> "DateTime":
        utc_datetime = datetime.strptime(datetime_str, DATETIME_FORMAT).replace(
            tzinfo=timezone.utc
        )
        return cls(utc_timestamp=utc_datetime.timestamp())

    def __str__(self) -> str:
        utc_datetime = datetime.fromtimestamp(self.utc_timestamp, tz=timezone.utc)
        return utc_datetime.strftime(DATETIME_FORMAT)

    def __hash__(self) -> int:
        return hash(self.utc_timestamp)

    def __sub__(self, other: "DateTime") -> timedelta:
        res = self.timedelta(other)
        return res

    def __eq__(self, other: Any) -> bool:
        if other is None:
            return False
        return self.utc_timestamp == other.utc_timestamp

    def __lt__(self, other: Self) -> bool:
        return self.utc_timestamp < other.utc_timestamp

    def timedelta(self, other: "DateTime") -> timedelta:
        utc_timestamp_delta = self.utc_timestamp - other.utc_timestamp
        return timedelta(seconds=utc_timestamp_delta)

    @classmethod
    def from_timestamp(cls, ts: float) -> datetime:
        return cls(utc_timestamp=ts)

    @classmethod
    def from_datetime(cls, dt: datetime) -> "DateTime":
        utc_datetime = dt.astimezone(timezone.utc)
        return cls(utc_timestamp=utc_datetime.timestamp())


def format_timedelta(local_timedelta: timedelta) -> str:
    total_seconds = int(local_timedelta.total_seconds())
    hours, leftover = divmod(total_seconds, 3600)
    minutes, seconds = divmod(leftover, 60)

    hours_string = f"{hours}:" if hours != 0 else ""
    minutes_string = f"{minutes}:".zfill(3)
    seconds_string = f"{seconds}".zfill(2)

    return f"{hours_string}{minutes_string}{seconds_string}"


def format_timedelta_human_readable(local_timedelta: timedelta) -> str:
    # Returns a human-readable string representing the timedelta
    units = [("day", 86400), ("hour", 3600), ("minute", 60), ("second", 1)]
    total_seconds = int(local_timedelta.total_seconds())

    for unit_name, unit_seconds in units:
        unit_value, total_seconds = divmod(total_seconds, unit_seconds)
        if unit_value > 0:
            if unit_value == 1:
                return f"{unit_value} {unit_name}"
            else:
                return f"{unit_value} {unit_name}s"
    return "0 seconds"
