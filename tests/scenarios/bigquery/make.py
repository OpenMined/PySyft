# third party
from fixtures_sync import create_user
from unsync import unsync

# syft absolute
import syft as sy
from syft import test_settings


@unsync
async def create_users(root_client, events, users, event_name):
    for test_user in users:
        create_user(root_client, test_user)
    events.register(event_name)


@unsync
async def create_endpoints_query(events, client, worker_pool_name: str, register: str):
    @sy.api_endpoint_method(
        settings={
            "credentials": test_settings.gce_service_account.to_dict(),
            "region": test_settings.gce_region,
            "project_id": test_settings.gce_project_id,
        }
    )
    def private_query_function(
        context,
        sql_query: str,
    ) -> str:
        # third party
        from google.cloud import bigquery  # noqa: F811
        from google.oauth2 import service_account

        # syft absolute
        from syft.service.response import SyftError

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

        try:
            rows = client.query_and_wait(
                sql_query,
                project=context.settings["project_id"],
            )

            if rows.total_rows > 1_000_000:
                return SyftError(
                    message="Please only write queries that gather aggregate statistics"
                )

            return rows.to_dataframe()
        except Exception as e:
            # We MUST handle the errors that we want to be visible to the data owners.
            # Any exception not catched is visible only to the data owner.
            # not a bigquery exception
            if not hasattr(e, "_errors"):
                output = f"got exception e: {type(e)} {str(e)}"
                return SyftError(
                    message=f"An error occured executing the API call {output}"
                )
                # return SyftError(message="An error occured executing the API call, please contact the domain owner.")

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
                return SyftError(
                    message="Error occured during the call: " + e._errors[0]["message"]
                )
            else:
                return SyftError(
                    message="An error occured executing the API call, please contact the domain owner."
                )

    # Define any helper methods for our rate limiter
    def is_within_rate_limit(context):
        """Rate limiter for custom API calls made by users."""
        # stdlib
        import datetime

        state = context.state
        settings = context.settings
        email = context.user.email

        current_time = datetime.datetime.now()
        calls_last_min = [
            1 if (current_time - call_time).seconds < 60 else 0
            for call_time in state[email]
        ]

        return sum(calls_last_min) < settings["CALLS_PER_MIN"]

    # Define a mock endpoint that the researchers can use for testing
    @sy.api_endpoint_method(
        settings={
            "credentials": test_settings.gce_service_account.to_dict(),
            "region": test_settings.gce_region,
            "project_id": test_settings.gce_project_id,
            "CALLS_PER_MIN": 10,
        },
        helper_functions=[is_within_rate_limit],
    )
    def mock_query_function(
        context,
        sql_query: str,
    ) -> str:
        # stdlib
        import datetime

        # third party
        from google.cloud import bigquery  # noqa: F811
        from google.oauth2 import service_account

        # syft absolute
        from syft.service.response import SyftError

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
        if context.user.email not in context.state.keys():
            context.state[context.user.email] = []

        if not context.code.is_within_rate_limit(context):
            return SyftError(message="Rate limit of calls per minute has been reached.")

        try:
            context.state[context.user.email].append(datetime.datetime.now())

            rows = client.query_and_wait(
                sql_query,
                project=context.settings["project_id"],
            )

            if rows.total_rows > 1_000_000:
                return SyftError(
                    message="Please only write queries that gather aggregate statistics"
                )

            return rows.to_dataframe()

        except Exception as e:
            # not a bigquery exception
            if not hasattr(e, "_errors"):
                output = f"got exception e: {type(e)} {str(e)}"
                return SyftError(
                    message=f"An error occured executing the API call {output}"
                )
                # return SyftError(message="An error occured executing the API call, please contact the domain owner.")

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
                return SyftError(
                    message="Error occured during the call: " + e._errors[0]["message"]
                )
            else:
                return SyftError(
                    message="An error occured executing the API call, please contact the domain owner."
                )

    new_endpoint = sy.TwinAPIEndpoint(
        path="bigquery.test_query",
        description="This endpoint allows to query Bigquery storage via SQL queries.",
        private_function=private_query_function,
        mock_function=mock_query_function,
        worker_pool=worker_pool_name,
    )

    result = client.custom_api.add(endpoint=new_endpoint)

    if register:
        if isinstance(result, sy.SyftSuccess):
            events.register(register)
        else:
            print("Failed to add api endpoint")
