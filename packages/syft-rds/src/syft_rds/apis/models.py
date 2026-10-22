"""The on-disk format of an api, shared by the DO (syft-bg) and the client.

<syftbox>/<do_email>/app_data/apis/<name>/
    api.yaml        # ApiDefinition
    files/<path>    # pinned copies of the content-matched files
    policies/input/run.sh (+ policy.py)    # optional input policy
    policies/output/run.sh (+ policy.py)   # optional output policy
    policy_state/   # what the policies keep between runs, data owner only
    syft.pub.yaml   # read access for the peers that can call the api
"""

import hashlib
from pathlib import Path

import yaml
from typing import Optional

from pydantic import BaseModel, Field

from syft_rds.apis.args import ApiArg

APIS_DIR = Path("app_data") / "apis"
API_FILE_NAME = "api.yaml"
FILES_DIR_NAME = "files"
RUN_SCRIPT_PATH = "run.sh"
# The DO writes the config.yaml of each job a call becomes, so no api pins it
# or takes it from a call.
JOB_CONFIG_PATH = "config.yaml"
PARAMS_FILE_NAME = "params.json"
POLICIES_DIR_NAME = "policies"
POLICY_STATE_DIR_NAME = "policy_state"


class FileEntry(BaseModel):
    """A pinned file of an api, with its hash."""

    relative_path: str  # e.g. "code/main.py"
    # Absolute path of the stored copy, filled in on load, e.g.
    # "<datasite>/app_data/apis/my_analysis/files/code/main.py"
    path: str = Field(default="", exclude=True)
    hash: str  # e.g. "sha256:abc123..."

    @classmethod
    def from_file(cls, relative_path: str, path: str | Path) -> "FileEntry":
        """Create a FileEntry from an existing file, computing its hash."""
        p = Path(path)
        content = p.read_text(encoding="utf-8")
        file_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        return cls(relative_path=relative_path, path=str(p), hash=file_hash)


class ApiDefinition(BaseModel):
    """Content-matched files, name-only files, peers and argument names."""

    file_contents: list[FileEntry] = Field(default_factory=list)
    file_paths: list[str] = Field(default_factory=list)
    peers: list[str] = Field(default_factory=list)
    args: list[ApiArg] = Field(default_factory=list)  # the params json keys
    # Every file of each policy folder, e.g. "policies/input/run.sh", hashed.
    input_policy: list[FileEntry] = Field(default_factory=list)
    output_policy: list[FileEntry] = Field(default_factory=list)
    # The items in DisclosureItem that a call's approval releases, e.g. "logs".
    disclosures: list[str] = Field(default_factory=list)

    @property
    def call_files(self) -> list[str]:
        """Every file a call sends: the files the api takes by name."""
        return list(self.file_paths)

    @property
    def params_file(self) -> Optional[str]:
        """The file client.api writes a call's arguments to, e.g. "code/params.json".

        See params_file_of.
        """
        return params_file_of(self.file_paths)


def params_file_of(file_paths: list[str]) -> Optional[str]:
    """The params file of an api that takes these files by name, or None.

    By convention, an api has arguments only when it takes one file by name
    and that file is a params.json.
    """
    if len(file_paths) == 1 and Path(file_paths[0]).name == PARAMS_FILE_NAME:
        return file_paths[0]
    return None


def definition_error(definition: ApiDefinition) -> Optional[str]:
    """Why the definition makes no api, or None."""
    checks = (
        _pins_run_script,
        _leaves_out_job_config,
        _pins_or_names_each_file_once,
        _has_params_file_for_args,
    )
    return next((error for check in checks if (error := check(definition))), None)


def _pinned(definition: ApiDefinition) -> set[str]:
    return {e.relative_path for e in definition.file_contents}


def _pins_run_script(definition: ApiDefinition) -> Optional[str]:
    if RUN_SCRIPT_PATH not in _pinned(definition):
        return f"An api pins {RUN_SCRIPT_PATH} by content"
    return None


def _leaves_out_job_config(definition: ApiDefinition) -> Optional[str]:
    """The DO writes config.yaml for each call."""
    if JOB_CONFIG_PATH in _pinned(definition) | set(definition.file_paths):
        return f"{JOB_CONFIG_PATH} is written for each call; leave it out"
    return None


def _pins_or_names_each_file_once(definition: ApiDefinition) -> Optional[str]:
    """A call sends exactly the name-only files, so none is also pinned."""
    both = _pinned(definition) & set(definition.file_paths)
    if both:
        return f"A file can't be pinned and taken by name: {sorted(both)}"
    return None


def _has_params_file_for_args(definition: ApiDefinition) -> Optional[str]:
    if definition.args and definition.params_file is None:
        return f"An api with arguments takes one file by name: its {PARAMS_FILE_NAME}"
    return None


def load_api_definition(api_dir: Path) -> ApiDefinition:
    """Load api.yaml, pointing each FileEntry.path at its stored copy."""
    data = yaml.safe_load((api_dir / API_FILE_NAME).read_text()) or {}
    definition = ApiDefinition.model_validate(data)
    for entry in definition.file_contents:
        entry.path = str(api_dir / FILES_DIR_NAME / entry.relative_path)
    for entry in definition.input_policy + definition.output_policy:
        entry.path = str(api_dir / entry.relative_path)
    return definition


def list_api_names(apis_dir: Path) -> list[str]:
    """Names of the apis in apis_dir, sorted.

    A folder counts once it holds api.yaml. Staging folders (".staging_*")
    hold one before they are renamed into place, so hidden folders are skipped.
    """
    if not apis_dir.exists():
        return []
    return sorted(
        d.name
        for d in apis_dir.iterdir()
        if d.is_dir() and not d.name.startswith(".") and (d / API_FILE_NAME).exists()
    )


def policy_dir(api_dir: Path, kind: str) -> Path:
    """Where the "input" or "output" policy of an api lives."""
    return api_dir / POLICIES_DIR_NAME / kind
