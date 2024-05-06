# stdlib
import enum
import json

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
    INFO = "info.svg"

    @property
    def svg(self) -> str:
        return load_svg(self.value)

    @property
    def js_escaped_svg(self) -> str:
        """
        Required for in-line SVGs in JS
        TODO remove after refactoring table + templating
        """
        return json.dumps(self.svg)
