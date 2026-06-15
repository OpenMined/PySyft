from pydantic import BaseModel
from typing import Dict, List, Callable
from pathlib import Path


class FileWriter(BaseModel):
    base_path: Path
    callbacks: Dict[str, List[Callable]] = {}
    write_files: bool = True

    def add_callback(self, on: str, callback: Callable):
        if on not in self.callbacks:
            self.callbacks[on] = []
        self.callbacks[on].append(callback)

    def write_file(self, path: str, content: str):
        target_path = self.base_path / path
        resolved_path = target_path.resolve()
        resolved_base = self.base_path.resolve()
        if not str(resolved_path).startswith(str(resolved_base)):
            raise ValueError(f"Path {path} is outside of base_path {self.base_path}")
        if self.write_files:
            with open(resolved_path, "w") as f:
                f.write(content)

        for callback in self.callbacks.get("write_file", []):
            callback(path, content)
