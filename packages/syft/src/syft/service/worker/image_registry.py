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

    @classmethod
    def from_url(cls, full_str: str):
        return cls(id=UID(), url=full_str)

    @property
    def tls_enabled(self) -> bool:
        return self.url.startswith("https")

    def __hash__(self) -> int:
        return hash(self.url + str(self.tls_enabled))

    def __repr__(self) -> str:
        return f"SyftImageRegistry(url={self.url})"

    def __str__(self) -> str:
        return self.url
