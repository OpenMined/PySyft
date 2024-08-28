# stdlib
from secrets import token_hex
import uuid

# third party
from faker import Faker
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.request.request import Request
from syft.service.request.request import UserCodeStatusChange
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.user.user import User
from syft.service.user.user_roles import ServiceRole
from syft.types.errors import SyftException


@sy.api_endpoint_method()
def private_query_function(context, query_str: str) -> str:
    return query_str


@sy.api_endpoint_method()
def mock_query_function(context, query_str: str) -> str:
    return query_str


@sy.api_endpoint(path="test.submit_query", endpoint_timeout=5)
def submit_query(
    context,
    func_name: str,
    query: str,
) -> str:
    # syft absolute
    import syft as sy

    @sy.syft_function(
        name=func_name,
        input_policy=sy.MixedInputPolicy(
            endpoint=sy.Constant(val=context.admin_client.api.services.test.test_query),
            query=sy.Constant(val=query),
            client=context.admin_client,
        ),
    )
    def execute_query(query: str, endpoint):
        res = endpoint(query_str=query)
        return res

    request = context.user_client.code.request_code_execution(execute_query)
    if isinstance(request, sy.SyftError):
        return request
    context.admin_client.requests.set_tags(request, ["autosync"])

    return (
        f"Query submitted {request}. Use `client.code.{func_name}()` to run your query"
    )


@pytest.mark.local_server
def test_compound_api() -> None:
    server = sy.orchestra.launch(
        name=token_hex(8),
        dev_mode=True,
        reset=True,
        local_db=True,
        create_producer=True,
        n_consumers=1,
    )
    root_client = server.login(email="info@openmined.org", password="changethis")
    new_endpoint = sy.TwinAPIEndpoint(
        path="test.test_query",
        description="Test",
        private_function=private_query_function,
        mock_function=mock_query_function,
    )

    res = root_client.custom_api.add(endpoint=new_endpoint)
    print(res)

    res = root_client.custom_api.add(endpoint=submit_query)
    print(res)
    res = root_client.users.create(
        email="newuser@openmined.org", name="John Doe", password="pw"
    )
    ds_client = root_client.login(email="newuser@openmined.org", password="pw")
    ds_client.api.services.test.submit_query(func_name="func_test", query="test query")
    root_client.requests[0].approve()
    assert ds_client.code.func_test().get() == "test query"
    server.python_server.cleanup()
    server.land()
