# stdlib
from pathlib import Path
import tomllib

# third party
from filelock import FileLock

FILE_LOCK_TIMEOUT = 30


class TomlReaderWriter:
    def __init__(self, lock: FileLock, filename: Path | str) -> None:
        self.filename = Path(filename)
        self.timeout = FILE_LOCK_TIMEOUT
        self.lock = lock

    def write(self, toml_dict: dict) -> None:
        with self.lock.acquire(timeout=self.timeout):
            with open(str(self.filename), "wb") as fp:
                tomllib.dump(toml_dict, fp)

    def read(self) -> dict:
        with self.lock.acquire(timeout=self.timeout):
            with open(str(self.filename), "rb") as fp:
                toml = tomllib.load(fp)
        return toml
