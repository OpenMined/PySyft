# stdlib
from collections.abc import Callable

# syft absolute
import syft as sy

# relative
from ..... import test_settings
from ..rate_limiter import is_within_rate_limit


def make_schema(settings: dict, worker_pool_name: str) -> Callable:
    updated_settings = {
        "calls_per_min": 5,
        "rate_limiter_enabled": True,
        "credentials": test_settings.gce_service_account.to_dict(),
        "region": test_settings.gce_region,
        "project_id": test_settings.gce_project_id,
        "dataset_1": test_settings.dataset_1,
        "table_1": test_settings.table_1,
        "table_2": test_settings.table_2,
    } | settings

    @sy.api_endpoint(
        path="bigquery.schema",
        description="This endpoint allows for visualising the metadata of tables available in BigQuery.",
        settings=updated_settings,
        helper_functions=[
            is_within_rate_limit
        ],  # Adds ratelimit as this is also a method available to data scientists
        worker_pool_name=worker_pool_name,
    )
    def live_schema(
        context,
    ) -> str:
        # stdlib
        import datetime

        # third party
        from google.cloud import bigquery  # noqa: F811
        from google.oauth2 import service_account
        import pandas as pd

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
            # Formats the data schema in a data frame format
            # Warning: the only supported format types are primitives, np.ndarrays and pd.DataFrames

            data_schema = []
            for table_id in [
                f"{context.settings['dataset_1']}.{context.settings['table_1']}",
                f"{context.settings['dataset_1']}.{context.settings['table_2']}",
            ]:
                table = client.get_table(table_id)
                for schema in table.schema:
                    data_schema.append(
                        {
                            "project": str(table.project),
                            "dataset_id": str(table.dataset_id),
                            "table_id": str(table.table_id),
                            "schema_name": str(schema.name),
                            "schema_field": str(schema.field_type),
                            "description": str(table.description),
                            "num_rows": str(table.num_rows),
                        }
                    )
            return pd.DataFrame(data_schema)

        except Exception as e:
            # not a bigquery exception
            if not hasattr(e, "_errors"):
                output = f"got exception e: {type(e)} {str(e)}"
                raise SyftException(
                    public_message=f"An error occured executing the API call {output}"
                )

            # Should add appropriate error handling for what should be exposed to the data scientists.
            raise SyftException(
                public_message="An error occured executing the API call, please contact the domain owner."
            )

    return live_schema
