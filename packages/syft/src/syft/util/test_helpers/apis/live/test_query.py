# stdlib
from collections.abc import Callable

# syft absolute
import syft as sy

# relative
from ..... import test_settings
from ..rate_limiter import is_within_rate_limit


def make_test_query(settings) -> Callable:
    updated_settings = {
        "calls_per_min": 10,
        "rate_limiter_enabled": True,
        "credentials": test_settings.gce_service_account.to_dict(),
        "region": test_settings.gce_region,
        "project_id": test_settings.gce_project_id,
    } | settings

    # these are the same if you allow the rate limiter to be turned on and off
    @sy.api_endpoint_method(
        settings=updated_settings,
        helper_functions=[is_within_rate_limit],
    )
    def live_test_query(
        context,
        sql_query: str,
    ) -> str:
        # stdlib
        import datetime

        # third party
        from google.cloud import bigquery  # noqa: F811
        from google.oauth2 import service_account

        # syft absolute
        from syft import SyftException

        # Auth for Bigquer based on the workload identity
        credentials = service_account.Credentials.from_service_account_info(
            context.settings["credentials"]
        )
        scoped_credentials = credentials.with_scopes(
            ["https://www.googleapis.com/auth/cloud-platform"]
        )

        client = bigquery.Client(
            credentials=scoped_credentials,
            location=context.settings["region"],
        )

        # Store a dict with the calltimes for each user, via the email.
        if context.settings["rate_limiter_enabled"]:
            if context.user.email not in context.state.keys():
                context.state[context.user.email] = []

            if not context.code.is_within_rate_limit(context):
                raise SyftException(
                    public_message="Rate limit of calls per minute has been reached."
                )
            context.state[context.user.email].append(datetime.datetime.now())

        try:
            rows = client.query_and_wait(
                sql_query,
                project=context.settings["project_id"],
            )

            if rows.total_rows > 1_000_000:
                raise SyftException(
                    public_message="Please only write queries that gather aggregate statistics"
                )

            return rows.to_dataframe()

        except Exception as e:
            # not a bigquery exception
            if not hasattr(e, "_errors"):
                output = f"got exception e: {type(e)} {str(e)}"
                raise SyftException(
                    public_message=f"An error occured executing the API call {output}"
                )

            # Treat all errors that we would like to be forwarded to the data scientists
            # By default, any exception is only visible to the data owner.

            if e._errors[0]["reason"] in [
                "badRequest",
                "blocked",
                "duplicate",
                "invalidQuery",
                "invalid",
                "jobBackendError",
                "jobInternalError",
                "notFound",
                "notImplemented",
                "rateLimitExceeded",
                "resourceInUse",
                "resourcesExceeded",
                "tableUnavailable",
                "timeout",
            ]:
                raise SyftException(
                    public_message="Error occured during the call: "
                    + e._errors[0]["message"]
                )
            else:
                raise SyftException(
                    public_message="An error occured executing the API call, please contact the domain owner."
                )

    return live_test_query
