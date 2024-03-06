# stdlib
from collections.abc import Callable

# relative
from ...types.transforms import TransformContext


def _downgrade_metadata_v3_to_v2() -> Callable:
    def set_defaults_from_settings(context: TransformContext) -> TransformContext:
        # Extract from settings if node is attached to context
        if context.output is not None:
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
