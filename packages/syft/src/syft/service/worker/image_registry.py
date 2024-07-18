# stdlib
import re

# third party
from pydantic import field_validator
from typing_extensions import Self

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject

# Checks for
# - localhost:[port]
# - (sub.)*.name.tld
# - (sub.)*.name.tld:[port]
REGX_DATASITE = re.compile(
    r"^(localhost|([a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*))(\:\d{1,5})?"
)


@serializable()
class SyftImageRegistry(SyftObject):
    __canonical_name__ = "SyftImageRegistry"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["url"]
    __attr_unique__ = ["url"]
    __repr_attrs__ = ["url"]

    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, val: str) -> str:
        if not val:
            raise ValueError("Invalid Registry URL. Must not be empty")

        if not bool(re.match(REGX_DATASITE, val)):
            raise ValueError("Invalid Registry URL. Must be a valid datasite.")

        return val

    @classmethod
    def from_url(cls, full_str: str) -> Self:
        return cls(url=full_str)

    def __hash__(self) -> int:
        return hash(self.url + str(self.tls_enabled))

    def __repr__(self) -> str:
        return f"SyftImageRegistry(url={self.url})"

    def __str__(self) -> str:
        return self.url
