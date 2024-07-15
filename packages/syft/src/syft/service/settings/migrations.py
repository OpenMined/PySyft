# stdlib
from collections.abc import Callable

# relative
from ...types.transforms import TransformContext


def set_from_server_to_key(server_attr: str, key: str) -> Callable:
    def extract_from_server(context: TransformContext) -> TransformContext:
        if context.output is not None:
            context.output[key] = getattr(context.server, server_attr)
        return context

    return extract_from_server
