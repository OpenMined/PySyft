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
from pydantic import BaseModel, Field

from syft_rds.apis.args import ApiArg

APIS_DIR = Path("app_data") / "apis"
API_FILE_NAME = "api.yaml"
FILES_DIR_NAME = "files"
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
    # The DO's syft-bg can run a call sent as a request (params only). A DS
    # client sends a whole job to an api written by an older syft-bg.
    accepts_requests: bool = False


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
