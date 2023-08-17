# stdlib
import mimetypes
import os
from pathlib import Path
import tempfile
from typing import Any
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

# extra mime types not in mimetypes.types_map
mime_types = {
    ".msgpack": "application/msgpack",
    ".yml": "application/yaml",
    ".yaml": "application/yaml",
}


def get_mimetype(filename: str, path: str) -> str:
    try:
        mimetype = mimetypes.guess_type(path)[0]
        if mimetype is not None:
            return mimetype

        extension_parts = filename.split(".")
        if len(extension_parts) > 1:
            extension = f".{extension_parts[-1]}"
            if extension in mime_types:
                return mime_types[extension]
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

    @property
    def name(self):
        return self.filename

    def head(self, length: int = 200) -> str:
        print(self.decode(length=length))

    def decode(self, length: int = -1) -> str:
        output = ""
        try:
            if length < 0:
                length = self.size_bytes
            slice_size = min(length, self.size_bytes)

            output = self.data[:slice_size].decode("utf-8")
        except Exception:  # nosec
            print("Failed to slice bytes")
        if length < self.size_bytes:
            output += "\n..."
        return output

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
    def from_string(
        content: str, filename: str = "file.txt", mime_type: Optional[str] = None
    ) -> Optional[Self]:
        data = content.encode("utf-8")
        if mime_type is None:
            mime_type = get_mimetype(filename, filename)
        return SyftFile(
            filename=filename,
            size_bytes=len(data),
            mimetype=mime_type,
            data=data,
        )

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
    # files: List[Union[SyftFile, SyftFolder]]
    files: List[Any]
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
            print(syft_file.name)
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
                        # syft_file = SyftFile.from_path(entry.path)
                        syft_files.append(syft_file)
                    except Exception:
                        print(f"Failed to load: {entry} as syft file")
                if entry.is_dir():
                    try:
                        print(entry.path.split("/")[-1])
                        syft_folder = SyftFolder.from_dir(
                            name=entry.path.split("/")[-1], path=entry.path
                        )
                        print(syft_folder.name)
                        syft_files.append(syft_folder)
                    except Exception as e:
                        print(e)
                        print(f"Failed to load: {entry} as syft folder: {entry.path}")

        return SyftFolder(name=name, files=syft_files)

    def write_file(self, path):
        try:
            full_path = f"{path}/{self.name}"
            print(full_path)
            if os.path.exists(full_path):
                return SyftSuccess(message=f"File already exists: {full_path}")
            # with open(full_path, "wb") as f:
            #     f.write(self.data)
            os.makedirs(full_path, exist_ok=True)
            print("HERE????")
            for syft_file in self.files:
                print(syft_file.name)
                syft_file.write_file(full_path)
            return SyftSuccess(message=f"File saved: {full_path}")
        except Exception as e:
            return SyftError(message=f"Failed to write {type(self)} to {path}. {e}")
