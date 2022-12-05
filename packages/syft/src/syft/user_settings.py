# future
from __future__ import annotations

# stdlib
import os

# third party
from dotenv import load_dotenv

# relative
from .core.common.decorators import singleton

# Get root path of project
root_path = os.path.dirname(os.path.abspath(__file__))

# get the .env file path
env_path = os.path.join(root_path, ".env")

# load the .env file
load_dotenv(dotenv_path=env_path)

# TODO: Add this to a file in ~/.syft and add some ENV overrides
tutorial_mode = type(os.environ.get("SYFT_TUTORIAL_MODE", True))
print(tutorial_mode)


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
