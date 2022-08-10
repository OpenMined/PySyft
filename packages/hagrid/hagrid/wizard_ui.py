# stdlib
from typing import Dict
from typing import Union

# relative
from .deps import Dependency
from .deps import check_grid_docker
from .deps import check_hagrid
from .deps import check_syft
from .deps import check_syft_deps
from .nb_output import NBOutput


class WizardUI:
    @property
    def check_hagrid(self) -> Union[Dict[str, Dependency], NBOutput]:
        return check_hagrid()

    @property
    def check_syft_deps(self) -> Union[Dict[str, Dependency], NBOutput]:
        return check_syft_deps()

    @property
    def check_syft(self) -> Union[Dict[str, Dependency], NBOutput]:
        return check_syft()

    @property
    def check_syft_pre(self) -> Union[Dict[str, Dependency], NBOutput]:
        return check_syft(pre=True)

    @property
    def check_grid_docker(self) -> Union[Dict[str, Dependency], NBOutput]:
        return check_grid_docker()
