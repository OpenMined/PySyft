# stdlib
from typing import Any

# relative
from ...types.errors import SyftException
from ...types.result import as_result
from ...util.util import get_mb_serialized_size
from ..metadata.server_metadata import ServerMetadata
from ..metadata.server_metadata import ServerMetadataJSON


def min_size_for_blob_storage_upload(
    metadata: ServerMetadata | ServerMetadataJSON,
) -> int:
    return metadata.min_size_blob_storage_mb


@as_result(SyftException)
def can_upload_to_blob_storage(
    data: Any, metadata: ServerMetadata | ServerMetadataJSON
) -> bool:
    try:
        return get_mb_serialized_size(data) >= min_size_for_blob_storage_upload(
            metadata
        )
    except TypeError as exc:
        raise SyftException.from_exception(exc, public_message=str(exc))
