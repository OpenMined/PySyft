# stdlib
import uuid

# third party
from faker import Faker
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.server.worker import Worker
from syft.service.action.action_endpoint import CustomEndpointActionObject
from syft.service.action.action_object import ActionObject
from syft.service.policy.policy import Constant, CreatePolicyRuleConstant
from syft.service.request.request import Request
from syft.service.request.request import UserCodeStatusChange
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.user.user import User
from syft.service.user.user_roles import ServiceRole
from syft.types.errors import SyftException

@sy.api_endpoint_method()
def private_query_function(
    context,
    query_str: str
) -> str:
    return query_str

@sy.api_endpoint_method()
def mock_query_function(
    context,
    query_str: str
) -> str:
    return query_str

def test_constant(worker, ds_client) -> None:
    
    root_client = worker.root_client
    new_endpoint = sy.TwinAPIEndpoint(
        path="test.test_query",
        description="Test",
        private_function=private_query_function,
        mock_function=mock_query_function,
    )
    
    res = root_client.custom_api.add(endpoint=new_endpoint)
    
    assert isinstance(res, SyftSuccess)
    
    create_constant = CreatePolicyRuleConstant(val=2)
    constant = create_constant.to_policy_rule("test")
    
    assert constant.val == 2
    assert constant.klass == int
    
    create_constant = CreatePolicyRuleConstant(val=root_client.api.services.test.test_query)
    constant = create_constant.to_policy_rule("test_2")
    
    assert constant.val == root_client.api.services.test.test_query.endpoint_id
    assert constant.klass == CustomEndpointActionObject
    
# def test_mixed_policy(worker, ds_client) -> None
