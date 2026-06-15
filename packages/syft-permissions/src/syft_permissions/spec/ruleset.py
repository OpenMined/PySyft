from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from syft_permissions.spec.rule import Rule

PERMISSION_FILE_NAME = "syft.pub.yaml"


class RuleSet(BaseModel):
    rules: list[Rule] = []
    terminal: bool = False
    path: str = Field(default="", exclude=True)

    @classmethod
    def load(cls, filepath: Path) -> "RuleSet":
        with open(filepath) as f:
            data = yaml.safe_load(f) or {}
        rs = cls.model_validate(data)
        rs.path = str(filepath.parent)
        return rs

    def save(self, filepath: Path | None = None) -> None:
        target = filepath or Path(self.path) / PERMISSION_FILE_NAME
        data = self.model_dump(mode="json")
        with open(target, "w") as f:
            yaml.safe_dump(data, f, default_flow_style=False)
