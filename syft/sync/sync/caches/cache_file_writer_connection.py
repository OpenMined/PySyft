from typing import Generic, List, TypeVar
from pydantic import BaseModel, Field
from pathlib import Path
import shutil
from typing import Tuple

T = TypeVar("T")


def is_valid_cache_type(cache_type: type) -> bool:
    return cache_type in (str, bytes) or issubclass(cache_type, BaseModel)


def _serialize(self, content: T) -> bytes:
    if isinstance(content, bytes):
        return content
    elif isinstance(content, str):
        return content.encode("utf-8")
    elif isinstance(content, BaseModel):
        return content.model_dump_json().encode("utf-8")
    else:
        raise TypeError(f"Unsupported content type: {type(content)}")


def _deserialize(self, data: bytes, content_type: type[T] | None = None) -> T:
    if content_type is not None:
        return content_type.model_validate_json(data.decode("utf-8"))
    try:
        return data.decode("utf-8")
    except Exception:
        return data
    # if content_type is bytes:
    #     return data
    # elif content_type is str:
    #     return data.decode("utf-8")
    # elif issubclass(content_type, BaseModel):
    #     return content_type.model_validate_json(data.decode("utf-8"))
    # else:
    #     raise TypeError(f"Unsupported content type: {content_type}")


class KeySortedDict(dict):
    """Dict where the items are sorted by key str, like files on a filesystem (possible)"""

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # Re-sort after each insertion
        sorted_items = sorted(self.items())
        self.clear()
        self.update(sorted_items)


class CacheFileConnection(BaseModel, Generic[T]):
    class Config:
        arbitrary_types_allowed = True

    @classmethod
    def get_generic_type(cls) -> type[T]:
        generic_args = cls.__pydantic_generic_metadata__.get("args", ())
        if not generic_args:
            raise TypeError(f"No generic type found on {cls.__name__}")
        return generic_args[0]

    def model_post_init(self, __context):
        super().model_post_init(__context)

        # generic_type = self.get_generic_type()
        # if not is_valid_cache_type(generic_type):
        #     raise TypeError(
        #         f"Invalid cache type: {generic_type} for {self.__class__.__name__}. Must be str, bytes, or a Pydantic BaseModel."
        #     )


class InMemoryCacheFileConnection(CacheFileConnection[T]):
    sorted_files: KeySortedDict[Path, T] = Field(default_factory=KeySortedDict)

    def get_items(self) -> List[Tuple[Path, T]]:
        return list(self.sorted_files.items())

    def clear_cache(self):
        self.sorted_files = KeySortedDict()

    def write_file(self, path: str | Path, content: T):
        path = Path(path)
        self.sorted_files[path] = content

    def read_file(self, path: str | Path) -> T:
        path = Path(path)
        return self.sorted_files[path]

    def delete_file(self, path: str | Path) -> None:
        path = Path(path)
        if path in self.sorted_files:
            del self.sorted_files[path]

    def delete_directory(self, path: str | Path) -> None:
        path = Path(path)
        keys_to_delete = [
            k for k in self.sorted_files if k == path or path in k.parents
        ]
        for k in keys_to_delete:
            del self.sorted_files[k]

    def __len__(self) -> int:
        return len(self.sorted_files)

    def __getitem__(self, idx: int) -> T:
        if not isinstance(idx, int):
            raise TypeError(f"Key must be an integer, got {type(idx)}")
        return list(self.sorted_files.values())[idx]

    def get_latest(self) -> T:
        return list(self.sorted_files.values())[-1]

    def get_all(self) -> List[T]:
        return list(self.sorted_files.values())


class FSFileConnection(CacheFileConnection[T]):
    base_dir: Path
    dtype: type[T] | None = None

    def get_items(self) -> List[Tuple[str, T]]:
        return [
            (
                str(f.relative_to(self.base_dir)),
                self._read_file_full_path(f),
            )
            for f in self._iter_files()
        ]

    def clear_cache(self):
        if self.base_dir.exists():
            for file_or_folder in self.base_dir.iterdir():
                if file_or_folder.is_file():
                    file_or_folder.unlink()
                elif file_or_folder.is_dir():
                    shutil.rmtree(file_or_folder)

    def get_keys(self) -> List[str]:
        return [str(f.relative_to(self.base_dir)) for f in self._iter_files()]

    def model_post_init(self, context):
        super().model_post_init(context)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _resolve_full_path(self, path: str) -> Path:
        # Convert to Path and resolve relative to base_dir
        full_path = (self.base_dir / path).resolve()
        base_dir_resolved = self.base_dir.resolve()

        # Ensure the path is within base_dir (prevent path traversal)
        if not full_path.is_relative_to(base_dir_resolved):
            raise ValueError(
                f"Path {path} is outside of the base directory {self.base_dir}"
            )

        return full_path

    def _iter_files(self) -> List[Path]:
        files = [f for f in self.base_dir.rglob("*") if f.is_file()]
        return sorted(files)

    def write_file(self, path: str, content: T) -> None:
        full_path = self._resolve_full_path(path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        data_bytes = _serialize(self, content)
        with open(full_path, "wb") as f:
            f.write(data_bytes)

    def read_file(self, path: str) -> T:
        full_path = self._resolve_full_path(path)
        return self._read_file_full_path(full_path)

    def delete_file(self, path: str | Path) -> None:
        path = Path(path)
        full_path = self._resolve_full_path(path)
        if full_path.exists():
            full_path.unlink()

    def delete_directory(self, path: str | Path) -> None:
        path = Path(path)
        full_path = self._resolve_full_path(path)
        if full_path.exists() and full_path.is_dir():
            shutil.rmtree(full_path)

    def _read_file_full_path(self, full_path: Path) -> T:
        with open(full_path, "rb") as f:
            res_bytes = f.read()
        res = _deserialize(self, res_bytes, self.dtype)
        return res

    def __len__(self) -> int:
        return len(self._iter_files())

    def __getitem__(self, idx: int) -> T:
        if not isinstance(idx, int):
            raise TypeError(f"Key must be an integer, got {type(idx)}")
        files = self._iter_files()
        file_path = files[idx]
        return self.read_file(str(file_path.relative_to(self.base_dir)))

    def get_latest(self) -> T:
        files = self._iter_files()
        latest_file = files[-1]
        return self.read_file(str(latest_file.relative_to(self.base_dir)))

    def get_all(self) -> List[T]:
        files = self._iter_files()
        return [self.read_file(str(f.relative_to(self.base_dir))) for f in files]

    def to_dict(self) -> dict[str, T]:
        files = self._iter_files()
        return {
            str(f.relative_to(self.base_dir)): self.read_file(
                str(f.relative_to(self.base_dir))
            )
            for f in files
        }
