"""Whether a request may call an api.

A request is a call of an api: the DS's chosen files, such as code/params.json,
and no code. The DO's client checks it here before it turns the request into a
job with the api's code.
"""

import json
from pathlib import Path

from syft_permissions.spec.ruleset import PERMISSION_FILE_NAME

from syft_rds.apis.api import callable_layout
from syft_rds.apis.args import bind_args
from syft_rds.apis.models import (
    ApiDefinition,
)

RUN_SCRIPT_PATH = "run.sh"
REQUEST_CONFIG_PATH = "config.yaml"


def submission_files(root: Path) -> dict[str, Path]:
    """Every file in a submission as {path relative to its root: abs_path}.

    The root holds `code/`, `run.sh` and `config.yaml`. `run.sh` is the only
    file the runner executes, so it must be in this set. The exact name
    `syft.pub.yaml` is left out, because the permission layer writes it. Any
    other spelling stays in, and is refused as an extra file: an exclusion from
    this inventory is a file nobody reviews, so keep it as narrow as the writer.
    """
    files: dict[str, Path] = {}
    if root.exists():
        for f in root.rglob("*"):
            if f.is_file() and f.name != PERMISSION_FILE_NAME:
                files[f.relative_to(root).as_posix()] = f
    return files


def check_request(
    root: Path, definition: ApiDefinition, submitter: str
) -> tuple[bool, str]:
    """Whether the request at root may call the api. Returns (ok, reason).

    A request holds no code, so nothing is matched by content. It may hold only
    the api's name-only files, its params must fit the api's arguments, and
    the submitter must be one of the api's peers.
    """
    if definition.peers and submitter not in definition.peers:
        return False, f"the api is not shared with {submitter}"
    sent = set(submission_files(root)) - {REQUEST_CONFIG_PATH}
    not_allowed = sent - set(definition.file_paths)
    if not_allowed:
        return False, f"files the api doesn't take: {sorted(not_allowed)}"
    return _check_params(root, definition)


def _check_params(root: Path, definition: ApiDefinition) -> tuple[bool, str]:
    """Whether the request's params json binds to the api's arguments."""
    layout = callable_layout(definition)
    if layout is None:
        return False, "the api can't be called with arguments"
    try:
        params = json.loads((root / layout.params_file).read_text())
        bind_args("api", definition.args, (), params)
    except (OSError, ValueError, TypeError) as e:
        return False, f"params don't fit the api: {e}"
    return True, "ok"
