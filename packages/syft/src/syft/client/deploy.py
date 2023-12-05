# stdlib
from typing import Any

# relative
from ..service.response import SyftError


class InstallOrchestra:
    def launch(self, *args: Any, **kwargs: Any) -> None:
        return self.error()

    def error(self) -> Any:
        message = "Please install hagrid with `pip install -U hagrid`"
        return SyftError(message=message)

    def _repr_html_(self) -> str:
        return self.error()._repr_html_()


def import_orchestra() -> Any:
    try:
        # third party
        from hagrid import Orchestra

        return Orchestra

    except Exception as e:  # nosec
        print(e)
        pass
    return InstallOrchestra()


Orchestra = import_orchestra()
