# future
from __future__ import annotations


# alert-info, alert-warning, alert-success, alert-danger
class NBOutput:
    def __init__(self, raw_output: str) -> None:
        self.raw_output = raw_output

    def _repr_html_(self) -> str:
        return self.raw_output

    def to_html(self) -> NBOutput:
        self.raw_output = self.raw_output.replace("\n", "<br />")
        return self
