# stdlib
from collections.abc import Callable

# syft absolute
import syft as sy

# relative
from ..rate_limiter import is_within_rate_limit
from .data import schema_dict


def make_schema(settings, worker_pool) -> Callable:
    updated_settings = {
        "calls_per_min": 5,
        "rate_limiter_enabled": True,
        "schema_dict": schema_dict,
    } | settings

    @sy.api_endpoint(
        path="bigquery.schema",
        description="This endpoint allows for visualising the metadata of tables available in BigQuery.",
        settings=updated_settings,
        helper_functions=[is_within_rate_limit],
        worker_pool=worker_pool,
    )
    def mock_schema(
        context,
    ) -> str:
        # syft absolute
        from syft import SyftException

        # Store a dict with the calltimes for each user, via the email.
        if context.settings["rate_limiter_enabled"]:
            # stdlib
            import datetime

            if context.user.email not in context.state.keys():
                context.state[context.user.email] = []

            context.state[context.user.email].append(datetime.datetime.now())

            if not context.code.is_within_rate_limit(context):
                raise SyftException(
                    public_message="Rate limit of calls per minute has been reached."
                )

        # third party
        import pandas as pd

        df = pd.DataFrame(context.settings["schema_dict"])
        return df

    return mock_schema
