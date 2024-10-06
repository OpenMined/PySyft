# stdlib
from typing import Any

# syft absolute
import syft as sy
from syft.util.test_helpers.apis import make_schema
from syft.util.test_helpers.apis import make_test_query

# relative
from ..sim.core import SimulatorContext
from .utils import server_info

__all__ = ["bq_schema_endpoint", "bq_test_endpoint", "bq_submit_endpoint"]


def bq_schema_endpoint(
    ctx: SimulatorContext,
    admin_client: sy.DatasiteClient,
    worker_pool: str,
    path: str = "bigquery.schema",
):
    schema_function = make_schema(
        settings={
            "calls_per_min": 5,
        },
        worker_pool_name=worker_pool,
        path=path,
    )

    # Call admin_client.custom_api.add
    __create_endpoint(ctx, admin_client, schema_function, path)


def bq_test_endpoint(
    ctx: SimulatorContext,
    admin_client: sy.DatasiteClient,
    worker_pool: str,
    path="bigquery.test_query",
):
    private_query_function = make_test_query(
        settings={
            "rate_limiter_enabled": False,
        }
    )
    mock_query_function = make_test_query(
        settings={
            "rate_limiter_enabled": True,
            "calls_per_min": 10,
        }
    )

    test_endpoint = sy.TwinAPIEndpoint(
        path=path,
        description="This endpoint allows to query Bigquery storage via SQL queries.",
        private_function=private_query_function,
        mock_function=mock_query_function,
        worker_pool_name=worker_pool,
        endpoint_timeout=120,
    )

    # Call admin_client.custom_api.add
    __create_endpoint(ctx, admin_client, test_endpoint, path)


def bq_submit_endpoint(
    ctx: SimulatorContext,
    admin_client: sy.DatasiteClient,
    worker_pool: str,
    path="bigquery.submit_query",
):
    @sy.api_endpoint(
        path=path,
        description="API endpoint that allows you to submit SQL queries to run on the private data.",
        worker_pool_name=worker_pool,
        settings={"worker": worker_pool},
        endpoint_timeout=120,
    )
    def submit_query(
        context,
        func_name: str,
        query: str,
    ) -> str:
        # stdlib
        import hashlib

        # syft absolute
        import syft as sy

        hash_object = hashlib.new("sha256")
        hash_object.update(context.user.email.encode("utf-8"))
        func_name = func_name + "_" + hash_object.hexdigest()[:6]

        @sy.syft_function(
            name=func_name,
            input_policy=sy.MixedInputPolicy(
                endpoint=sy.Constant(
                    val=context.admin_client.api.services.bigquery.test_query
                ),
                query=sy.Constant(val=query),
                client=context.admin_client,
            ),
            worker_pool_name=context.settings["worker"],
        )
        def execute_query(query: str, endpoint):
            res = endpoint(sql_query=query)
            return res

        request = context.user_client.code.request_code_execution(execute_query)
        if isinstance(request, sy.SyftError):
            return request
        context.admin_client.requests.set_tags(request, ["autosync"])

        return f"Query submitted {request}. Use `client.code.{func_name}()` to run your query"

    # Call admin_client.custom_api.add
    __create_endpoint(ctx, admin_client, submit_query, path)


def __create_endpoint(
    ctx: SimulatorContext,
    admin_client: sy.DatasiteClient,
    endpoint: Any,
    path: str,
):
    msg = f"Admin {admin_client.metadata.server_side_type}: Endpoint '{path}' on {server_info(admin_client)}"
    ctx.logger.info(f"{msg} - Creating")

    # Create the endpoint
    result = admin_client.custom_api.add(endpoint=endpoint)
    assert isinstance(result, sy.SyftSuccess), result

    ctx.logger.info(f"{msg} - Created")
