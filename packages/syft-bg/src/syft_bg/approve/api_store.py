"""Storage for auto-approval objects ("apis") inside the DO's SyftBox datasite.

One folder per api, not partitioned by user because an api can be called by
several peers. The format is defined in syft_rds.apis.models, so clients can
read the apis shared with them.
"""

import hashlib
import shutil
import tempfile
from pathlib import Path

import yaml
from syft_permissions import PERMISSION_FILE_NAME, Access, Rule, RuleSet
from syft_rds.apis.models import (
    API_FILE_NAME,
    APIS_DIR,
    FILES_DIR_NAME,
    load_api_definition,
)

from syft_bg.approve.config import AutoApprovalObj, FileEntry
from syft_bg.common.config import get_syftbg_dir
from syft_bg.common.locking import file_lock

EVERYONE = "*"


class ApiExistsError(Exception):
    """Raised when an api folder can't be moved into place."""


class ApiStore:
    """Reads and writes auto-approval objects under app_data/apis/."""

    def __init__(self, syftbox_root: str | Path, do_email: str):
        self.datasite = Path(syftbox_root).expanduser() / do_email
        self.apis_dir = self.datasite / APIS_DIR

    def api_dir(self, name: str) -> Path:
        return self.apis_dir / name

    def names(self) -> list[str]:
        if not self.apis_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.apis_dir.iterdir()
            if d.is_dir() and (d / API_FILE_NAME).exists()
        )

    def load_all(self) -> dict[str, AutoApprovalObj]:
        return {name: self.get(name) for name in self.names()}

    def get(self, name: str) -> AutoApprovalObj:
        """Load an api, pointing each FileEntry.path at its stored copy."""
        return load_api_definition(self.api_dir(name))

    def exists(self, name: str) -> bool:
        return (self.api_dir(name) / API_FILE_NAME).exists()

    def create(
        self,
        name: str,
        content_files: list[tuple[str, Path]],
        file_paths: list[str],
        peers: list[str],
        args: list[str] | None = None,
    ) -> tuple[str, AutoApprovalObj]:
        """Store a new api. Returns the final (unique) name and the object.

        Files are copied into a staging dir first, outside the lock, so the
        lock only covers picking a free name and the rename into place.
        """
        self.apis_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging_", dir=self.apis_dir))
        try:
            entries = _copy_and_hash_files(content_files, staging / FILES_DIR_NAME)
            obj = AutoApprovalObj(
                file_contents=entries,
                file_paths=file_paths,
                peers=peers,
                args=args or [],
            )
            _write_api_file(staging, obj)
            with file_lock(_lock_path()):
                name = unique_name(name, set(self.names()))
                _move_into_place(staging, self.api_dir(name))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self._write_permissions(name, peers)
        return name, self.get(name)

    def save(self, name: str, obj: AutoApprovalObj) -> None:
        """Overwrite an existing api's metadata and permissions."""
        _write_api_file(self.api_dir(name), obj)
        self._write_permissions(name, obj.peers)

    def delete(self, name: str) -> bool:
        api_dir = self.api_dir(name)
        if not api_dir.exists():
            return False
        shutil.rmtree(api_dir, ignore_errors=True)
        return True

    def _write_permissions(self, name: str, peers: list[str]) -> None:
        """Grant read on the whole api folder to its peers, or everyone.

        The file is rewritten as a whole, so dropped peers lose access.
        """
        readers = sorted(set(peers)) or [EVERYONE]
        ruleset = RuleSet(rules=[Rule(pattern="**", access=Access(read=readers))])
        ruleset.save(self.api_dir(name) / PERMISSION_FILE_NAME)


def unique_name(name: str, existing: set[str]) -> str:
    """Return name, or name_<n> with the lowest free n if it's taken."""
    if name not in existing:
        return name
    counter = 1
    while f"{name}_{counter}" in existing:
        counter += 1
    return f"{name}_{counter}"


def _lock_path() -> Path:
    # Kept out of SyftBox so the lock file never syncs to peers.
    lock = get_syftbg_dir() / "apis.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    return lock


def _move_into_place(staging: Path, final_dir: Path) -> None:
    try:
        staging.rename(final_dir)
    except OSError as e:
        raise ApiExistsError(f"Could not create api '{final_dir.name}': {e}") from e


def _write_api_file(api_dir: Path, obj: AutoApprovalObj) -> None:
    data = obj.model_dump(mode="json")
    (api_dir / API_FILE_NAME).write_text(yaml.safe_dump(data, sort_keys=False))


def _copy_and_hash_files(
    content_files: list[tuple[str, Path]], files_dir: Path
) -> list[FileEntry]:
    """Copy files into files_dir and compute their hashes."""
    entries: list[FileEntry] = []
    for rel_path, abs_path in content_files:
        dest = files_dir / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(abs_path, dest)
        content = dest.read_text(encoding="utf-8")
        file_hash = "sha256:" + hashlib.sha256(content.encode("utf-8")).hexdigest()
        entries.append(
            FileEntry(relative_path=rel_path, path=str(dest), hash=file_hash)
        )
    return entries
