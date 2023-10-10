# future
from __future__ import annotations

# stdlib
import os

# relative
from ..util.decorators import singleton

# TODO: Add this to a file in ~/.syft and add some ENV overrides
tutorial_mode = bool(os.environ.get("SYFT_TUTORIAL_MODE", True))


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
