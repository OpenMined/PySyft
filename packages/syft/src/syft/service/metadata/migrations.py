# stdlib
from typing import Callable

# relative
from ...types.syft_migration import migrate
from ...types.transforms import TransformContext
from ...types.transforms import drop
from ...types.transforms import rename
from .node_metadata import NodeMetadata
from .node_metadata import NodeMetadataV2
from .node_metadata import NodeMetadataV3


@migrate(NodeMetadata, NodeMetadataV2)
def upgrade_metadata_v1_to_v2():
    return [
        rename("highest_object_version", "highest_version"),
        rename("lowest_object_version", "lowest_version"),
    ]


@migrate(NodeMetadataV2, NodeMetadata)
def downgrade_metadata_v2_to_v1():
    return [
        rename("highest_version", "highest_object_version"),
        rename("lowest_version", "lowest_object_version"),
    ]


@migrate(NodeMetadataV2, NodeMetadataV3)
def upgrade_metadata_v2_to_v3():
    return [drop(["deployed_on", "on_board", "signup_enabled", "admin_email"])]


def _downgrade_metadata_v3_to_v2() -> Callable:
    def set_defaults_from_settings(context: TransformContext) -> TransformContext:
        # Extract from settings if node is attached to context
        if context.node is not None:
            context.output["deployed_on"] = context.node.settings.deployed_on
            context.output["on_board"] = context.node.settings.on_board
            context.output["signup_enabled"] = context.node.settings.signup_enabled
            context.output["admin_email"] = context.node.settings.admin_email
        else:
            # Else set default value
            context.output["signup_enabled"] = False
            context.output["admin_email"] = ""

        return context

    return set_defaults_from_settings


@migrate(NodeMetadataV3, NodeMetadataV2)
def downgrade_metadata_v3_to_v2():
    return [_downgrade_metadata_v3_to_v2()]
