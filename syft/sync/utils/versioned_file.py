"""Read JSON that carries a format version."""

import json
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

ModelT = TypeVar("ModelT", bound=BaseModel)


def parse_versioned_json(
    data: str | bytes, model: type[ModelT], max_version: int, kind: str
) -> ModelT:
    """``data`` as ``model``. ``kind`` names the data in errors.

    The version is checked before the fields, so data of a later layout is
    refused by its version, not by a field error.

    Raises:
        ValueError: if a later client wrote the data, or it has another shape.
    """
    version = json.loads(data)["version"]
    if version > max_version:
        raise ValueError(
            f"The {kind} has version {version}, and this client reads up to "
            f"version {max_version}."
        )
    return model.model_validate_json(data)


def read_versioned_json(
    path: Path, model: type[ModelT], max_version: int, kind: str
) -> ModelT:
    """The file at ``path`` as ``model``. ``kind`` names the file in errors.

    Raises:
        ValueError: if a later client wrote the file, or it has another shape.
    """
    return parse_versioned_json(
        path.read_text(), model, max_version, f"{kind} at {path}"
    )
