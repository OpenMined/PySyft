# stdlib
from datetime import datetime

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from .syft_object import SYFT_OBJECT_VERSION_1
from .syft_object import SyftObject


@serializable()
class DateTime(SyftObject):
    __canonical_name__ = "DateTime"
    __version__ = SYFT_OBJECT_VERSION_1

    utc_timestamp: float

    @staticmethod
    def now() -> Self:
        return DateTime(utc_timestamp=datetime.utcnow().timestamp())

    def __str__(self) -> str:
        utc_datetime = datetime.utcfromtimestamp(self.utc_timestamp)
        return utc_datetime.strftime("%Y-%m-%d %H:%M:%S")
