# stdlib
import json
from pathlib import Path
from tempfile import gettempdir

# third party
from filelock import FileLock


class SharedState:
    """A simple class to manage a file-backed shared state between multiple processes, particulary for pytest-xdist."""

    def __init__(self, name: str):
        self._dir = Path(gettempdir(), name)
        self._dir.mkdir(parents=True, exist_ok=True)

        self._statefile = Path(self._dir, "state.json")
        self._statefile.touch()

        self._lock = FileLock(str(self._statefile) + ".lock")

    @property
    def lock(self):
        return self._lock

    def set(self, key, value):
        with self._lock:
            state = self.read_state()
            state[key] = value
            self.write_state(state)
            return value

    def get(self, key, default=None):
        with self._lock:
            state = self.read_state()
            return state.get(key, default)

    def read_state(self) -> dict:
        return json.loads(self._statefile.read_text() or "{}")

    def write_state(self, state):
        self._statefile.write_text(json.dumps(state))

    def purge(self):
        self._statefile.unlink()
        Path(self._lock.lock_file).unlink()
