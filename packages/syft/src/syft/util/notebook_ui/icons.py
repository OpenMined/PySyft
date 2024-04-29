# stdlib
import enum

# relative
from ..assets import load_svg


class Icon(enum.Enum):
    SEARCH = "search.svg"
    CLIPBOARD = "clipboard.svg"
    TABLE = "table.svg"
    FOLDER = "folder.svg"
    REQUEST = "request.svg"
    ARROW = "arrow.svg"
    COPY = "copy.svg"

    @property
    def svg(self) -> str:
        return load_svg(self.value)
