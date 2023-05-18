# stdlib
import mimetypes
import os
from pathlib import Path
import tempfile
from typing import List
from typing import Union
from typing import Any
from typing import Callable
from typing import Optional
# third party
from typing_extensions import Self

# relative
from ..serde.serializable import serializable
from ..service.response import SyftError
from ..service.response import SyftSuccess
from ..types.syft_object import SYFT_OBJECT_VERSION_1
from ..types.syft_object import SyftObject
from .transforms import generate_id
from .transforms import transform
from .uid import UID
from .transforms import TransformContext

mime_types = {
    "txt": "text/plain",
    "json": "application/json",
    "msgpack": "application/msgpack",
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
    id: UID
    filename: str
    size_bytes: int
    mimetype: str

    __attr_repr_cols__ = ["filename", "mimetype", "size_bytes"]


@serializable()
class CreateSyftFile(SyftObject):
    __canonical_name__ = "CreateSyftFile"
    __version__ = SYFT_OBJECT_VERSION_1
    id: Optional[UID] = None
    filename: str
    data: bytes
    size_bytes: int
    mimetype: str

    __attr_repr_cols__ = ["filename", "mimetype", "size_bytes"]

    @staticmethod
    def write_file(filename: str, data: bytes, path: str) -> Union[SyftSuccess, SyftError]:
        try:
            full_path = f"{path}/{filename}"
            if os.path.exists(full_path):
                return SyftSuccess(message=f"File already exists: {full_path}")
            with open(full_path, "wb") as f:
                f.write(data)
            return SyftSuccess(message=f"File saved: {full_path}")
        except Exception as e:
            return SyftError(message=f"Failed to write SyftFile to {path}. {e}")    
        
    @staticmethod
    def from_path(path: str) -> Self:
        filename = os.path.basename(path)
        try:
            with open(path, "rb") as f:
                file_size = os.path.getsize(path)
                syft_file = CreateSyftFile(
                    filename=filename,
                    size_bytes=file_size,
                    mimetype=get_mimetype(filename, path),
                    data=f.read(),
                )
                return syft_file
        except Exception:
            print(f"Failed to load: {path} as syft file")

def download_file(context: TransformContext) -> TransformContext:
    CreateSyftFile.write_file(
        filename=context.output["filename"],
        data=context.output["data"],
        path=context.output['path']
    )
    return context

@transform(CreateSyftFile, SyftFile)
def createsyftfile_to_syftfile() -> List[Callable]:
    return [generate_id, download_file]


@serializable()
class ModelConfig(SyftObject):
    __canonical_name__ = "ModelConfig"
    __version__ = SYFT_OBJECT_VERSION_1
    files: List[SyftFile]
    name: str
    type: str

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
        pass

    @staticmethod
    def from_dir():
        pass
    
    @staticmethod
    def from_model(model: Any) -> Self:
        pass
    
@serializable()
class HuggingFaceTransformerModel(ModelConfig):
    __canonical_name__ = "HuggingFaceTransformerModel"
    __version__ = SYFT_OBJECT_VERSION_1
    name: str
    files: List[SyftFile]
    type: str
    
    @staticmethod
    def from_model(model: Any) -> Self:
        return super().from_model(model)
    
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
        return HuggingFaceTransformerModel(name=name, files=syft_files)
    