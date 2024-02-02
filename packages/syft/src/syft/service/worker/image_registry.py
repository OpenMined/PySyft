# stdlib
from urllib.parse import urlparse

# third party
from pydantic import validator

# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SyftObject
from ...types.uid import UID


@serializable()
class SyftImageRegistry(SyftObject):
    __canonical_name__ = "SyftImageRegistry"
    __version__ = SYFT_OBJECT_VERSION_1

    __attr_searchable__ = ["url"]
    __attr_unique__ = ["url"]

    __repr_attrs__ = ["url"]

    id: UID
    url: str

    @validator("url")
    def validate_url(cls, val: str):
        if val.startswith("http") or "://" in val:
            raise ValueError("Registry URL must be a valid RFC 3986 URI")
        return val

    @classmethod
    def from_url(cls, full_str: str):
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
