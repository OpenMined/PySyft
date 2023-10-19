# stdlib
from typing import Callable

# relative
from ...types.syft_migration import migrate
from ...types.transforms import TransformContext
from ...types.transforms import drop
from .settings import NodeSettings
from .settings import NodeSettingsV2


def set_from_node_to_key(node_attr: str, key: str) -> Callable:
    def extract_from_node(context: TransformContext) -> TransformContext:
        context.output[key] = getattr(context.node, node_attr)
        return context

    return extract_from_node


@migrate(NodeSettings, NodeSettingsV2)
def upgrade_metadata_v1_to_v2():
    return [
        set_from_node_to_key("verify_key", "verify_key"),
        set_from_node_to_key("node_type", "node_type"),
    ]


@migrate(NodeSettingsV2, NodeSettings)
def downgrade_metadata_v2_to_v1():
    return [
        drop(["verify_key", "node_type"]),
    ]
