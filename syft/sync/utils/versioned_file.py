"""Read a JSON file that carries a format version."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def read_versioned_json(
    path: Path, model: type[ModelT], max_version: int, kind: str
) -> ModelT:
    """The file at ``path`` as ``model``. ``kind`` names the file in errors.

    The version is checked before the fields, so a file of a later layout is
    refused by its version, not by a field error.

    Raises:
        ValueError: if a later client wrote the file, or it has another shape.
    """
    text = path.read_text()
    version = json.loads(text)["version"]
    if version > max_version:
        raise ValueError(
            f"The {kind} at {path} has version {version}, and this client "
            f"reads up to version {max_version}."
        )
    return model.model_validate_json(text)
