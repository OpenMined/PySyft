# stdlib
from collections.abc import Callable

# relative
from ...types.transforms import TransformContext


def set_from_node_to_key(node_attr: str, key: str) -> Callable:
    def extract_from_node(context: TransformContext) -> TransformContext:
        if context.output is not None:
            context.output[key] = getattr(context.node, node_attr)
        return context

    return extract_from_node
