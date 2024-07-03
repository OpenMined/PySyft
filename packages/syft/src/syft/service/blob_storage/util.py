# stdlib
from typing import Any

# relative
from ...util.util import get_mb_serialized_size
from ..metadata.node_metadata import NodeMetadata
from ..metadata.node_metadata import NodeMetadataJSON


def min_size_for_blob_storage_upload(metadata: NodeMetadata | NodeMetadataJSON) -> int:
    return metadata.min_size_blob_storage_mb


def can_upload_to_blob_storage(
    data: Any, metadata: NodeMetadata | NodeMetadataJSON
) -> bool:
    return get_mb_serialized_size(data) >= min_size_for_blob_storage_upload(metadata)
