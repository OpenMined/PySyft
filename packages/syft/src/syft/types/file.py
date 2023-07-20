# stdlib
import mimetypes
import os
from pathlib import Path
import tempfile
from typing import List
from typing import Optional
from typing import Union

# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject

mime_types = {
    "txt": "text/plain",
    "json": "application/json",
    "msgpack": "application/msgpack",
    "jpg": "image/jpeg",
    "png": "image/png",
}


def get_mimetype(filename: str, path: str) -> str:
    try:
        mimetype = mimetypes.guess_type(path)[0]
        if mimetype is not None:
            return mimetype

        extension = filename.split(".")
        if len(extension) > 1 and extension[-1] in mime_types:
            return mime_types[extension[-1]]
    except Exception as e:
        print(f"failed to get mime type. {e}")
    return "application/octet-stream"


@serializable()
class SyftFile(SyftObject):
    __canonical_name__ = "SyftFile"
    __version__ = SYFT_OBJECT_VERSION_1
    filename: str
    data: bytes
    size_bytes: int
    mimetype: str

    __attr_repr_cols__ = ["filename", "mimetype", "size_bytes"]

    def write_file(self, path: str) -> Union[SyftSuccess, SyftError]:
        try:
            full_path = f"{path}/{self.filename}"
            if os.path.exists(full_path):
                return SyftSuccess(message=f"File already exists: {full_path}")
            with open(full_path, "wb") as f:
                f.write(self.data)
            return SyftSuccess(message=f"File saved: {full_path}")
        except Exception as e:
            return SyftError(message=f"Failed to write {type(self)} to {path}. {e}")

    @staticmethod
    def from_path(path: str) -> Optional[Self]:
        abs_path = os.path.abspath(os.path.expanduser(path))
        if os.path.exists(abs_path):
            try:
                with open(abs_path, "rb") as f:
                    file_size = os.path.getsize(abs_path)
                    filename = os.path.basename(abs_path)
                    return SyftFile(
                        filename=filename,
                        size_bytes=file_size,
                        mimetype=get_mimetype(filename, abs_path),
                        data=f.read(),
                    )
            except Exception:
                print(f"Failed to load: {path} as syft file")
        return None


@serializable()
class SyftFolder(SyftObject):
    __canonical_name__ = "SyftFolder"
    __version__ = SYFT_OBJECT_VERSION_1
    files: List[SyftFile]
    name: str

    @property
    def size_bytes(self) -> int:
        total_size = 0
        for syft_file in self.files:
            total_size += syft_file.size_bytes
        return total_size

    @property
    def size_mb(self) -> float:
        return self.size_bytes / 1024 / 1024

    @property
    def model_folder(self) -> str:
        # TODO: make sure filenames are sanitized and arent file paths
        # TODO: this path should be unique to the user so you cant hijack another folder
        path = Path(tempfile.gettempdir()) / self.name
        os.makedirs(path, exist_ok=True)
        for syft_file in self.files:
            syft_file.write_file(path)
        return path

    @staticmethod
    def from_dir(name: str, path: str, ignore_hidden: bool = True) -> Self:
        syft_files = []
        path = Path(os.path.abspath(os.path.expanduser(path)))
        if not os.path.exists(path):
            raise Exception(f"{path} does not exist")

        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_file():
                    if ignore_hidden and entry.name.startswith("."):
                        continue
                    try:
                        with open(entry.path, "rb") as f:
                            file_size = os.path.getsize(entry.path)
                            syft_file = SyftFile(
                                filename=entry.name,
                                size_bytes=file_size,
                                mimetype=get_mimetype(entry.name, entry.path),
                                data=f.read(),
                            )
                            syft_files.append(syft_file)
                    except Exception:
                        print(f"Failed to load: {entry} as syft file")
        return SyftFolder(name=name, files=syft_files)
