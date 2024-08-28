# stdlib
from contextlib import nullcontext as does_not_raise

# third party
import pytest

# syft absolute
import syft as sy
from syft.service.code.user_code import UserCode
from syft.service.request.request import Request
from syft.service.request.request import RequestStatus
from syft.service.response import SyftError
from syft.types.errors import SyftException


@sy.api_endpoint_method(settings={})
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
def get_clients(full_high_worker):
    admin_client = full_high_worker.login(
        email="info@openmined.org", password="changethis"
    )
    admin_client.register(
        email="ds_user@test.abc",
        name="ds_user",
        password="1234",
        password_verify="1234",
    )
    ds_client = full_high_worker.login(email="ds_user@test.abc", password="1234")
    yield admin_client, ds_client


@pytest.fixture
def setup_query_endpoint(get_clients):
    admin_client, ds_client = get_clients
    new_endpoint = sy.TwinAPIEndpoint(
        path="bigquery.test_query",
        description="This endpoint allows to query Bigquery storage via SQL queries.",
        private_function=private_query_function,
        mock_function=mock_query_function,
    )

    admin_client.custom_api.add(endpoint=new_endpoint)

    yield admin_client, ds_client


@pytest.fixture
def update_query_endpoint(setup_query_endpoint):
    admin_client, ds_client = setup_query_endpoint
    admin_client.api.services.api.update(
        endpoint_path="bigquery.test_query", hide_mock_definition=True
    )
    admin_client.api.services.api.update(
        endpoint_path="bigquery.test_query", endpoint_timeout=10
    )
    yield admin_client, ds_client


@pytest.fixture
def create_submit_query_endpoint(update_query_endpoint):
    admin_client, ds_client = update_query_endpoint

    admin_client.custom_api.add(endpoint=submit_query)
    admin_client.api.services.api.update(
        endpoint_path="bigquery.submit_query", hide_mock_definition=True
    )
    yield admin_client, ds_client


@pytest.fixture
def submit_ds_request(create_submit_query_endpoint):
    admin_client, ds_client = create_submit_query_endpoint
    FUNC_NAME = "my_test_function"
    QUERY = "SELECT * FROM `bigquery-public-data.ml_datasets.penguins` LIMIT 10"

    submit_res = ds_client.api.services.bigquery.submit_query(
        func_name=FUNC_NAME, query=QUERY
    )

    fn_name = extract_code_path(submit_res)
    yield admin_client, ds_client, fn_name


@pytest.fixture
def approve_original_request(submit_ds_request):
    admin_client, ds_client, fn_name = submit_ds_request

    request = admin_client.requests[0]

    request.approve()

    yield admin_client, ds_client, fn_name


@pytest.fixture
def accept_request_by_deposit(submit_ds_request):
    admin_client, ds_client, fn_name = submit_ds_request

    request = admin_client.requests[0]

    job = execute_request(admin_client, request)
    job.wait()
    job_info = job.info(result=True)
    request.deposit_result(job_info, approve=True)
    yield admin_client, ds_client, fn_name


@pytest.fixture
def reject_request(submit_ds_request):
    admin_client, ds_client, fn_name = submit_ds_request

    request = admin_client.requests[0]

    request.deny(reason="Bad vibes :(")

    yield admin_client, ds_client, fn_name


def extract_code_path(response):
    # stdlib
    import re

    pattern = r"client\.code\.(\w+)\(\)"
    match = re.search(pattern, str(response))
    if match:
        extracted_code = match.group(1)
        return extracted_code
    return None


def execute_request(client_high, request) -> dict:
    if not isinstance(request, Request):
        return "This is not a request"

    code = request.code
    if not isinstance(code, UserCode):
        return "No usercode found"

    func_name = request.code.service_func_name
    api_func = getattr(client_high.code, func_name, None)
    if api_func is None:
        return "Code name was not found on the client."

    job = api_func(blocking=False)
    return job


# set up public submit query endpoint
@pytest.mark.local_server
def test_query_endpoint_added(update_query_endpoint) -> None:
    admin_client, _ = update_query_endpoint
    assert len(admin_client.custom_api.api_endpoints()) == 1


@pytest.mark.local_server
def test_query_endpoint_mock_endpoint(update_query_endpoint) -> None:
    query = "SELECT * FROM dataset_1.table_1 LIMIT 10"
    admin_client, _ = update_query_endpoint

    mock_result = admin_client.api.services.bigquery.test_query.mock(sql_query=query)
    assert not isinstance(mock_result, SyftError)

    retrieved_obj = mock_result.get()

    assert isinstance(retrieved_obj, str)
    assert "MOCK QUERY" in retrieved_obj
    assert query in retrieved_obj


@pytest.mark.local_server
def test_query_endpoint_private_endpoint(update_query_endpoint) -> None:
    query = "SELECT * FROM dataset_1.table_1 LIMIT 100"
    admin_client, _ = update_query_endpoint

    result = admin_client.api.services.bigquery.test_query.private(sql_query=query)
    assert not isinstance(result, SyftError)

    retrieved_obj = result.get()

    assert isinstance(retrieved_obj, str)
    assert "PRIVATE QUERY" in retrieved_obj
    assert query in retrieved_obj


@pytest.mark.local_server
def test_submit_query_endpoint_added(create_submit_query_endpoint):
    admin_client, _ = create_submit_query_endpoint

    assert len(admin_client.custom_api.api_endpoints()) == 2


@pytest.mark.local_server
def test_submit_query_endpoint(create_submit_query_endpoint) -> None:
    admin_client, ds_client = create_submit_query_endpoint
    sql_query = "SELECT * FROM dataset_1.table_1 LIMIT 10"
    # Inspect the context state on an endpoint
    result = admin_client.api.services.bigquery.submit_query(
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

    result = getattr(admin_client.code, fn_to_call)()

    assert not isinstance(result, SyftError)

    retrieved_obj = result.get()

    assert isinstance(retrieved_obj, str)

    assert "PRIVATE QUERY" in retrieved_obj
    assert sql_query in retrieved_obj


@pytest.mark.local_server
@pytest.mark.parametrize(
    "fixture_name, expected_request_status, raises_expectation, error_message",
    [
        (
            "submit_ds_request",
            RequestStatus.PENDING,
            pytest.raises(SyftException),
            "UserCodeStatus.PENDING",
        ),
        ("approve_original_request", RequestStatus.APPROVED, does_not_raise(), ""),
        (
            "reject_request",
            RequestStatus.REJECTED,
            pytest.raises(SyftException),
            "Bad vibes :(",
        ),
        ("accept_request_by_deposit", RequestStatus.APPROVED, does_not_raise(), ""),
    ],
)
def test_ds_result_retrieval(
    fixture_name, expected_request_status, raises_expectation, error_message, request
):
    flow_fixture = request.getfixturevalue(fixture_name)
    admin_client, ds_client, fn_name = flow_fixture

    req = admin_client.requests[0]
    assert isinstance(req, Request)
    assert req.status == expected_request_status

    assert "my_test_function" in fn_name

    api_method = getattr(ds_client.code, fn_name)
    with raises_expectation as exc:
        api_method()
    if error_message:
        assert error_message in str(exc.value)
