import hashlib
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
        hash_path = filepath.with_suffix(filepath.suffix + ".sha256")
        if hash_path.exists():
            with open(hash_path) as hf:
                expected_hash = hf.read().strip()
            with open(filepath, "rb") as f:
                actual_hash = hashlib.sha256(f.read()).hexdigest()
            if actual_hash != expected_hash:
                raise ValueError(f"File integrity check failed for {filepath}")
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
        hash_data = hashlib.sha256(target.read_bytes()).hexdigest()
        with open(target.with_suffix(target.suffix + ".sha256"), "w") as hf:
            hf.write(hash_data)
