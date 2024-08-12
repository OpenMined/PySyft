# stdlib
from typing import Any

# relative
from ...service.response import SyftException
from ...util.util import get_mb_serialized_size
from ..metadata.server_metadata import ServerMetadata
from ..metadata.server_metadata import ServerMetadataJSON


def min_size_for_blob_storage_upload(
    metadata: ServerMetadata | ServerMetadataJSON,
) -> int:
    return metadata.min_size_blob_storage_mb


def can_upload_to_blob_storage(
    data: Any, metadata: ServerMetadata | ServerMetadataJSON
) -> bool:
    serialized_size = get_mb_serialized_size(data)
    if serialized_size.is_err():
        raise SyftException(f"{serialized_size.err()}")
    return serialized_size.ok() >= min_size_for_blob_storage_upload(metadata)
