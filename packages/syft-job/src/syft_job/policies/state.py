"""The state folder an api's policies keep between runs."""

import hashlib
import json
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from syft_permissions import PERMISSION_FILE_NAME, Access, Rule, RuleSet

from .locking import file_lock

DEFAULT_STATE_FILE_NAME = "state.json"
LOCKS_DIR_NAME = "syft-policy-locks"


class PolicyState:
    """state.json and any other files a policy writes, for one api."""

    def __init__(self, state_dir: Path):
        self.state_dir = Path(state_dir)

    @property
    def default_state_path(self) -> Path:
        return self.state_dir / DEFAULT_STATE_FILE_NAME

    def ensure(self) -> None:
        """Create the folder, and state.json as {} on the first run."""
        self.state_dir.mkdir(parents=True, exist_ok=True)
        if not self.default_state_path.exists():
            self.default_state_path.write_text("{}")

    def read_default(self) -> Any:
        try:
            return json.loads(self.default_state_path.read_text())
        except (OSError, ValueError):
            return None

    @contextmanager
    def lock(self) -> Iterator[None]:
        """Hold the state for a whole policy run, so runs never overlap.

        The lock file sits in the temp folder, outside the SyftBox, so it never
        syncs.
        """
        key = hashlib.sha256(str(self.state_dir.resolve()).encode()).hexdigest()
        lock_path = Path(tempfile.gettempdir()) / LOCKS_DIR_NAME / f"{key}.lock"
        with file_lock(lock_path):
            yield

    def restrict_to_owner(self) -> None:
        """Make the folder readable by the datasite owner only.

        The api folder around it is readable by the api's peers. A terminal
        rule here stops that rule from reaching the state.
        """
        self.state_dir.mkdir(parents=True, exist_ok=True)
        ruleset = RuleSet(rules=[Rule(pattern="**", access=Access())], terminal=True)
        ruleset.save(self.state_dir / PERMISSION_FILE_NAME)
