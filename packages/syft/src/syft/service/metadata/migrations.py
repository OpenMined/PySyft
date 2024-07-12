# stdlib
from collections.abc import Callable

# relative
from ...types.transforms import TransformContext


def _downgrade_metadata_v3_to_v2() -> Callable:
    def set_defaults_from_settings(context: TransformContext) -> TransformContext:
        # Extract from settings if server is attached to context
        if context.output is not None:
            if context.server is not None:
                context.output["deployed_on"] = context.server.settings.deployed_on
                context.output["on_board"] = context.server.settings.on_board
                context.output["signup_enabled"] = (
                    context.server.settings.signup_enabled
                )
                context.output["admin_email"] = context.server.settings.admin_email
            else:
                # Else set default value
                context.output["signup_enabled"] = False
                context.output["admin_email"] = ""

        return context

    return set_defaults_from_settings
