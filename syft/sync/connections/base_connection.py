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
    """How to log in to one platform. Each platform has its own subclass, such
    as ``GdriveConnectionConfig``."""
