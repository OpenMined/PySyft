# third party
from helpers.events import unsync

# syft absolute
import syft as sy
from syft import test_settings
from syft.util.test_helpers.apis import make_schema
from syft.util.test_helpers.apis import make_test_query


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


@unsync
async def create_endpoints_query(events, client, worker_pool_name: str, register: str):
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


@unsync
async def create_endpoints_schema(events, client, worker_pool_name: str, register: str):
    schema_function = make_schema(
        settings={
            "calls_per_min": 5,
        },
        worker_pool=worker_pool_name,
    )
    result = client.custom_api.add(endpoint=schema_function)

    if register:
        if isinstance(result, sy.SyftSuccess):
            events.register(register)
        else:
            print("Failed to add schema_function")


@unsync
async def create_endpoints_submit_query(
    events, client, worker_pool_name: str, register: str
):
    @sy.api_endpoint(
        path="bigquery.submit_query",
        description="API endpoint that allows you to submit SQL queries to run on the private data.",
        worker_pool=worker_pool_name,
        settings={"worker": worker_pool_name},
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

    result = client.custom_api.add(endpoint=submit_query)

    if register:
        if isinstance(result, sy.SyftSuccess):
            events.register(register)
        else:
            print("Failed to add api endpoint")


@unsync
async def set_endpoint_settings(
    events, client, path, kwargs, after: str, register: str
):
    if after:
        await events.await_for(event_name=after)

    # Here, we update the endpoint to timeout after 100s (rather the default of 60s)
    result = client.api.services.api.update(endpoint_path=path, **kwargs)
    if isinstance(result, sy.SyftSuccess):
        events.register(register)


def api_for_path(client, path):
    root = client.api.services
    for part in path.split("."):
        if hasattr(root, part):
            root = getattr(root, part)
    return root


def run_api_path(client, path, **kwargs):
    api_method = api_for_path(client, path)
    result = api_method(**kwargs)
    return result


def query_sql():
    dataset_2 = test_settings.get("dataset_2", default="dataset_2")
    table_2 = test_settings.get("table_2", default="table_2")
    table_2_col_id = test_settings.get("table_2_col_id", default="table_id")
    table_2_col_score = test_settings.get("table_2_col_score", default="colname")

    query = f"SELECT {table_2_col_id}, AVG({table_2_col_score}) AS average_score \
        FROM {dataset_2}.{table_2} \
        GROUP BY {table_2_col_id} \
        LIMIT 10000"
    return query
