from aries_cloudagent.config.injection_context import InjectionContext
from aries_cloudagent.core.protocol_registry import ProtocolRegistry
from .message_types import MESSAGE_TYPES


async def setup(
        context: InjectionContext,
        protocol_registry: ProtocolRegistry = None
):
    """Setup the protocolexample plugin."""
    if not protocol_registry:
        protocol_registry = await context.inject(ProtocolRegistry)
    protocol_registry.register_message_types(
        MESSAGE_TYPES
    )
