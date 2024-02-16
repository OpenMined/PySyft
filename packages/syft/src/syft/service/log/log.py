# relative
from ...serde.serializable import serializable
from ...types.syft_migration import migrate
from ...types.syft_object import SYFT_OBJECT_VERSION_1
from ...types.syft_object import SYFT_OBJECT_VERSION_2
from ...types.syft_object import SyftObject
from ...types.transforms import drop
from ...types.transforms import make_set_default


@serializable()
class SyftLogV1(SyftObject):
    __canonical_name__ = "SyftLog"
    __version__ = SYFT_OBJECT_VERSION_1

    stdout: str = ""

    def append(self, new_str: str) -> None:
        self.stdout += new_str


@serializable()
class SyftLog(SyftObject):
    __canonical_name__ = "SyftLog"
    __version__ = SYFT_OBJECT_VERSION_2

    stdout: str = ""
    stderr: str = ""

    def append_error(self, new_str: str) -> None:
        self.stderr += new_str

    def restart(self) -> None:
        self.stderr = ""
        self.stdout = ""


@migrate(SyftLogV1, SyftLog)
def upgrade_syftlog_v1_to_v2():
    return [
        make_set_default("stderr", ""),
    ]


@migrate(SyftLog, SyftLogV1)
def downgrade_syftlog_v2_to_v1():
    return [
        drop("stderr"),
    ]
