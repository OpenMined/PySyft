from typing import ClassVar, Type
from pydantic import BaseModel


class FileCollection(BaseModel):
    """A collection of files that can be shared and synced."""

    folder_id: str
    tag: str
    content_hash: str
    has_any_permission: bool = False
    # The layout this collection holds, as written into the folder name. An owner
    # publishes one collection per layout its audience reads; "" is the original
    # layout. See CollectionSyncSpec.
    variant: str = ""


class ConnectionConfig(BaseModel):
    connection_type: ClassVar[Type["SyftboxPlatformConnection"]]


class SyftboxPlatformConnection(BaseModel):
    config: ConnectionConfig | None = None

    @classmethod
    def from_config(cls, config: ConnectionConfig):
        return config.connection_type.from_config(config)
