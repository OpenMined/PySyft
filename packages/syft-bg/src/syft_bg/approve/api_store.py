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
from syft_job.policies import (
    PYTHON_POLICY_NAME,
    RUN_SCRIPT_NAME,
    PolicyKind,
    PolicyState,
    python_run_script,
)
from syft_job.policies.locking import file_lock
from syft_permissions import PERMISSION_FILE_NAME, Access, Rule, RuleSet
from syft_rds.apis import ApiArg
from syft_rds.apis.models import (
    API_FILE_NAME,
    APIS_DIR,
    FILES_DIR_NAME,
    POLICY_STATE_DIR_NAME,
    list_api_names,
    load_api_definition,
    policy_dir,
)

from syft_bg.approve.config import AutoApprovalObj, FileEntry
from syft_bg.common.config import get_syftbg_dir

EVERYONE = "*"

# Passed to set_policies for a policy that stays as it is.
UNCHANGED = object()


class ApiExistsError(Exception):
    """Raised when an api folder can't be moved into place."""


class DuplicateApiError(ApiExistsError):
    """Raised when a new api matches the same jobs as an existing one."""


class ApiNotFoundError(KeyError):
    """Raised when no api has the given name."""


class ApiStore:
    """Reads and writes auto-approval objects under app_data/apis/."""

    def __init__(self, syftbox_root: str | Path, do_email: str):
        self.datasite = Path(syftbox_root).expanduser() / do_email
        self.apis_dir = self.datasite / APIS_DIR

    def api_dir(self, name: str) -> Path:
        return self.apis_dir / name

    def names(self) -> list[str]:
        return list_api_names(self.apis_dir)

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
        args: list[ApiArg] | None = None,
        input_policy: Path | None = None,
        output_policy: Path | None = None,
    ) -> tuple[str, AutoApprovalObj]:
        """Store a new api. Returns the final (unique) name and the object.

        Files are copied into a staging dir first, outside the lock, so the
        lock only covers the duplicate check, picking a free name and the
        rename into place. A policy is a .sh file (stored as run.sh) or a .py
        file (stored as policy.py next to a generated run.sh).

        Raises DuplicateApiError when an api with the same pinned and
        name-only files exists: a job could then match both.
        """
        self.apis_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".staging_", dir=self.apis_dir))
        try:
            obj = AutoApprovalObj(
                file_contents=_copy_and_hash_files(
                    content_files, staging / FILES_DIR_NAME
                ),
                file_paths=file_paths,
                peers=peers,
                args=args or [],
                input_policy=store_policy(staging, PolicyKind.INPUT, input_policy),
                output_policy=store_policy(staging, PolicyKind.OUTPUT, output_policy),
                accepts_requests=True,
            )
            _write_api_file(staging, obj)
            with file_lock(_lock_path()):
                self._refuse_duplicate(obj)
                name = unique_name(name, set(self.names()))
                _move_into_place(staging, self.api_dir(name))
        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)
            raise
        self._write_permissions(name, peers)
        self._create_policy_state(name)
        return name, self.get(name)

    def set_policies(
        self,
        name: str,
        input_policy: Path | None | object = UNCHANGED,
        output_policy: Path | None | object = UNCHANGED,
        reset_state: bool = False,
    ) -> AutoApprovalObj:
        """Add, replace or remove (None) the policies of an existing api.

        The state is kept unless reset_state is True.
        """
        if not self.exists(name):
            raise ApiNotFoundError(f"No api named '{name}'")
        api_dir = self.api_dir(name)
        obj = self.get(name)
        with file_lock(_lock_path()):
            if input_policy is not UNCHANGED:
                obj.input_policy = store_policy(api_dir, PolicyKind.INPUT, input_policy)
            if output_policy is not UNCHANGED:
                obj.output_policy = store_policy(
                    api_dir, PolicyKind.OUTPUT, output_policy
                )
            _write_api_file(api_dir, obj)
        if reset_state:
            shutil.rmtree(api_dir / POLICY_STATE_DIR_NAME, ignore_errors=True)
        self._create_policy_state(name)
        return self.get(name)

    def _refuse_duplicate(self, obj: AutoApprovalObj) -> None:
        """Refuse an api that could match the same job as an existing one."""
        signature = _match_signature(obj)
        for existing in self.names():
            other = self.get(existing)
            if _match_signature(other) == signature and _peers_overlap(obj, other):
                raise DuplicateApiError(
                    f"Api '{existing}' already matches these files. Use "
                    f"syft_bg.set_policies('{existing}', ...) to change its "
                    f"policies instead."
                )

    def _create_policy_state(self, name: str) -> None:
        state = PolicyState(self.api_dir(name) / POLICY_STATE_DIR_NAME)
        state.ensure()
        state.restrict_to_owner()

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


def _match_signature(obj: AutoApprovalObj) -> tuple[frozenset, frozenset]:
    """What decides which jobs an api matches: its pinned and name-only files."""
    pinned = frozenset((e.relative_path, e.hash) for e in obj.file_contents)
    return pinned, frozenset(obj.file_paths)


def _peers_overlap(a: AutoApprovalObj, b: AutoApprovalObj) -> bool:
    """Whether some peer can call both; no peers means any peer."""
    return not a.peers or not b.peers or bool(set(a.peers) & set(b.peers))


def store_policy(
    api_dir: Path, kind: PolicyKind, source: Path | None
) -> list[FileEntry]:
    """Replace the kind's policy folder with source; None removes it."""
    folder = policy_dir(api_dir, kind.value)
    shutil.rmtree(folder, ignore_errors=True)
    if source is None:
        return []
    source = Path(source).expanduser()
    folder.mkdir(parents=True)
    if source.suffix == ".py":
        shutil.copy2(source, folder / PYTHON_POLICY_NAME)
        (folder / RUN_SCRIPT_NAME).write_text(python_run_script())
    elif source.suffix == ".sh":
        shutil.copy2(source, folder / RUN_SCRIPT_NAME)
    else:
        raise ValueError(f"A policy must be a .py or .sh file, got {source}")
    return [
        FileEntry.from_file(f.relative_to(api_dir).as_posix(), f)
        for f in sorted(folder.iterdir())
    ]


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
