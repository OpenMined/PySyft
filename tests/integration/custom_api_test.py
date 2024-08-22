# stdlib

# third party
import pytest

# syft absolute
import syft as sy
from syft.service.response import SyftError

secrets = {
    "service_account_bigquery_private": {},
    "service_account_bigquery_mock": {},
    "region_bigquery": "",
    "project_id": "",
    "dataset_1": "dataset1",
    "table_1": "table1",
    "table_2": "table2",
}
# use fixtures to setup low server, set up high server

# fixture to optionally set up worker pool (or use default)


# create custom endpoint for private, mock query functions on high server
@sy.api_endpoint_method(
    settings={
        "credentials": secrets["service_account_bigquery_private"],
        "region": secrets["region_bigquery"],
        "project_id": secrets["project_id"],
    }
)
def private_query_function(
    context,
    sql_query: str,
) -> str:
    return f"PRIVATE QUERY: {sql_query}"


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


@sy.api_endpoint_method(
    settings={
        "credentials": secrets["service_account_bigquery_private"],
        "region": secrets["region_bigquery"],
        "project_id": secrets["project_id"],
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

    # syft absolute
    from syft.service.response import SyftError

    # Store a dict with the calltimes for each user, via the email.
    if context.user.email not in context.state.keys():
        context.state[context.user.email] = []

    if not context.code.is_within_rate_limit(context):
        return SyftError(message="Rate limit of calls per minute has been reached.")

    try:
        context.state[context.user.email].append(datetime.datetime.now())

        return f"MOCK QUERY: {sql_query}"

    except Exception:
        return SyftError(
            message="An error occured executing the API call, please contact the domain owner."
        )


@sy.api_endpoint(
    path="bigquery.submit_query",
    description="API endpoint that allows you to submit SQL queries to run on the private data.",
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
    )
    def execute_query(query: str, endpoint):
        res = endpoint(sql_query=query)
        return res

    request = context.user_client.code.request_code_execution(execute_query)
    if isinstance(request, sy.SyftError):
        return request
    context.admin_client.requests.set_tags(request, ["autosync"])

    return (
        f"Query submitted {request}. Use `client.code.{func_name}()` to run your query"
    )


@pytest.fixture
def setup_query_endpoint(full_high_worker):
    high_client = full_high_worker.login(
        email="info@openmined.org", password="changethis"
    )
    new_endpoint = sy.TwinAPIEndpoint(
        path="bigquery.test_query",
        description="This endpoint allows to query Bigquery storage via SQL queries.",
        private_function=private_query_function,
        mock_function=mock_query_function,
    )

    high_client.custom_api.add(endpoint=new_endpoint)

    yield high_client


@pytest.fixture
def update_query_endpoint(setup_query_endpoint):
    client = setup_query_endpoint
    client.api.services.api.update(
        endpoint_path="bigquery.test_query", hide_mock_definition=True
    )
    client.api.services.api.update(
        endpoint_path="bigquery.test_query", endpoint_timeout=10
    )
    yield client


# set up schema endpoint (not sure if this one is needed yet)


# set up public submit query endpoint
def test_query_endpoint_added(update_query_endpoint) -> None:
    high_client = update_query_endpoint
    assert len(high_client.custom_api.api_endpoints()) == 1


def test_query_endpoint_mock_endpoint(update_query_endpoint) -> None:
    query = f"SELECT * FROM {secrets['dataset_1']}.{secrets['table_1']} LIMIT 10"
    high_client = update_query_endpoint

    mock_result = high_client.api.services.bigquery.test_query.mock(
        sql_query=f"SELECT * FROM {secrets['dataset_1']}.{secrets['table_1']} LIMIT 10"
    )
    assert not isinstance(mock_result, SyftError)

    retrieved_obj = mock_result.get()

    assert isinstance(retrieved_obj, str)
    assert "MOCK QUERY" in retrieved_obj
    assert query in retrieved_obj


def test_query_endpoint_private_endpoint(update_query_endpoint) -> None:
    query = f"SELECT * FROM {secrets['dataset_1']}.{secrets['table_1']} LIMIT 10"
    high_client = update_query_endpoint

    result = high_client.api.services.bigquery.test_query.private(
        sql_query=f"SELECT * FROM {secrets['dataset_1']}.{secrets['table_1']} LIMIT 10"
    )
    assert not isinstance(result, SyftError)

    retrieved_obj = result.get()

    assert isinstance(retrieved_obj, str)
    assert "PRIVATE QUERY" in retrieved_obj
    assert query in retrieved_obj


@pytest.fixture
def create_submit_query_endpoint(update_query_endpoint):
    high_client = update_query_endpoint

    high_client.custom_api.add(endpoint=submit_query)
    high_client.api.services.api.update(
        endpoint_path="bigquery.submit_query", hide_mock_definition=True
    )
    yield high_client


def test_submit_query_endpoint_added(create_submit_query_endpoint):
    high_client = create_submit_query_endpoint

    assert len(high_client.custom_api.api_endpoints()) == 2


def test_submit_query_endpoint_endpoint(create_submit_query_endpoint) -> None:
    high_client = create_submit_query_endpoint
    sql_query = f"SELECT * FROM {secrets['dataset_1']}.{secrets['table_1']} LIMIT 1"
    # Inspect the context state on an endpoint
    result = high_client.api.services.bigquery.submit_query(
        func_name="my_func",
        query=sql_query,
    )

    assert not isinstance(result, SyftError)

    retrieved_obj = result.get()

    assert isinstance(retrieved_obj, str)
    assert "client.code.my_func" in retrieved_obj
    # stdlib
    import re

    fn_name_pattern = re.compile(r"client\.code\.(.*)\(")
    fn_to_call = fn_name_pattern.findall(retrieved_obj)[0]
    assert "my_func" in fn_to_call

    result = getattr(high_client.code, fn_to_call)()

    assert not isinstance(result, SyftError)

    retrieved_obj = result.get()

    assert isinstance(retrieved_obj, str)

    assert "PRIVATE QUERY" in retrieved_obj
    assert sql_query in retrieved_obj
