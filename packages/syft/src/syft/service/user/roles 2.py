# stdlib
from enum import Enum


class Roles(Enum):
    SOURCE_CONTRIBUTOR = "Source Contributor"
    UPLOADER = "Uploader"
    EDITOR = "Editor"

    def __str__(self):
        return f"{self.value}"
