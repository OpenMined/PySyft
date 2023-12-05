# relative
from ...serde.serializable import serializable
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject


@serializable()
class SyftLog(SyftObject):
    __canonical_name__ = "SyftLog"
    __version__ = SYFT_OBJECT_VERSION_1

    stdout: str = ""

    def append(self, new_str: str) -> None:
        self.stdout += new_str


@serializable()
class SyftLogV2(SyftLog):
    __canonical_name__ = "SyftLog"
    __version__ = SYFT_OBJECT_VERSION_2

    stderr: str = ""

    def append_error(self, new_str: str) -> None:
        self.stderr += new_str

    def restart(self) -> None:
        self.stderr = ""
        self.stdout = ""
