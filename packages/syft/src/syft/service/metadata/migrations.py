# relative
from ...types.syft_migration import migrate
from ...types.transforms import rename
from .node_metadata import NodeMetadata
from .node_metadata import NodeMetadataV1


@migrate(NodeMetadataV1, NodeMetadata)
def upgrade_metadata_v1_to_v2():
    return [
        rename("highest_object_version", "highest_version"),
        rename("lowest_object_version", "lowest_version"),
    ]


@migrate(NodeMetadata, NodeMetadataV1)
def downgrade_metadata_v2_to_v1():
    return [
        rename("highest_version", "highest_object_version"),
        rename("lowest_version", "lowest_object_version"),
    ]
