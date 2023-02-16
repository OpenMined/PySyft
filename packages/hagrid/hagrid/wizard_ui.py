# stdlib
from typing import Dict
from typing import Union

# relative
from .cache import arg_cache
from .deps import Dependency
from .deps import check_grid_docker
from .deps import check_hagrid
from .deps import check_syft
from .deps import check_syft_deps
from .nb_output import NBOutput

steps = {}
steps["check_hagrid"] = False
steps["check_syft"] = False
steps["check_grid"] = False


def complete_install_wizard(
    output: Union[Dict[str, Dependency], NBOutput]
) -> Union[Dict[str, Dependency], NBOutput]:
    flipped = arg_cache["install_wizard_complete"]
    if not flipped:
        for _, v in steps.items():
            if v is False:
                return output
    arg_cache["install_wizard_complete"] = True
    if isinstance(output, NBOutput):
        if flipped != arg_cache["install_wizard_complete"]:
            output.raw_output += "\n\n✅ You have completed the Install Wizard"
    return output


class WizardUI:
    @property
    def check_hagrid(self) -> Union[Dict[str, Dependency], NBOutput]:
        steps["check_hagrid"] = True
        return complete_install_wizard(check_hagrid())

    @property
    def check_syft_deps(self) -> Union[Dict[str, Dependency], NBOutput]:
        steps["check_syft"] = True
        return complete_install_wizard(check_syft_deps())

    @property
    def check_syft(self) -> Union[Dict[str, Dependency], NBOutput]:
        steps["check_syft"] = True
        return complete_install_wizard(check_syft())

    @property
    def check_syft_pre(self) -> Union[Dict[str, Dependency], NBOutput]:
        steps["check_syft"] = True
        return complete_install_wizard(check_syft(pre=True))

    @property
    def check_grid_docker(self) -> Union[Dict[str, Dependency], NBOutput]:
        steps["check_grid"] = True
        return complete_install_wizard(check_grid_docker())
