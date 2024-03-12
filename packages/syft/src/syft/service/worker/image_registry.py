# stdlib
import re
from urllib.parse import urlparse

# third party
from pydantic import field_validator
from typing_extensions import Self

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.uid import UID

REGX_DOMAIN = re.compile(r"^(localhost|([a-zA-Z0-9-]+(\.[a-zA-Z0-9-]+)*))(\:\d{1,5})?$")


@serializable()
class SyftImageRegistry(SyftObject):
    __canonical_name__ = "SyftImageRegistry"
    __version__ = SYFT_OBJECT_VERSION_2

    __attr_searchable__ = ["url"]
    __attr_unique__ = ["url"]

    __repr_attrs__ = ["url"]

    id: UID
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, val: str) -> str:
        if not val:
            raise ValueError("Invalid Registry URL. Must not be empty")

        if not bool(re.match(REGX_DOMAIN, val)):
            raise ValueError("Invalid Registry URL. Must be a valid domain.")

        return val

    @classmethod
    def from_url(cls, full_str: str) -> Self:
        # this is only for urlparse
        if "://" not in full_str:
            full_str = f"http://{full_str}"
        parsed = urlparse(full_str)

        # netloc includes the host & port, so local dev should work as expected
        return cls(id=UID(), url=parsed.netloc)

    def __hash__(self) -> int:
        return hash(self.url + str(self.tls_enabled))

    def __repr__(self) -> str:
        return f"SyftImageRegistry(url={self.url})"

    def __str__(self) -> str:
        return self.url
