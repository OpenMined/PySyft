# stdlib
import venv

# relative
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject


class Env(SyftObject):
    __canonical_name__ = "Env"
    __version__ = SYFT_OBJECT_VERSION_1
    packages_dict: dict[str, str]

    @property
    def packages(self) -> list[tuple[str, str]]:
        return list(self.packages_dict.items())

    def create_local_env(self) -> None:
        venv.EnvBuilder()
