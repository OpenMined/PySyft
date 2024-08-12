# third party

# relative
from ..serde.serializable import serializable
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject


@serializable()
class MarkdownDescription(SyftObject):
    # version
    __canonical_name__ = "MarkdownDescription"
    __version__ = SYFT_OBJECT_VERSION_1

    text: str

    def _repr_markdown_(self, wrap_as_python: bool = True, indent: int = 0) -> str:
        return self.text


@serializable()
class HTMLObject(SyftObject):
    # version
    __canonical_name__ = "HTMLObject"
    __version__ = SYFT_OBJECT_VERSION_1

    text: str

    def _repr_html_(self) -> str:
        return self.text
