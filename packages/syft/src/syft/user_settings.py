# future
from __future__ import annotations

# relative
from .core.common.decorators import singleton

# TODO: Add this to a file in ~/.syft and add some ENV overrides
tutorial_mode = True


@singleton
class UserSettings:
    def __init__(self) -> None:
        pass

    @property
    def helper(self) -> bool:
        return tutorial_mode

    @helper.setter
    def helper(self, value: bool) -> None:
        global tutorial_mode
        tutorial_mode = value


settings = UserSettings()
