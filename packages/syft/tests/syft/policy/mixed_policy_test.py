# stdlib

# third party

# syft absolute
import syft as sy
from syft.service.action.action_endpoint import CustomEndpointActionObject
from syft.service.action.action_object import ActionObject
from syft.service.policy.policy import CreatePolicyRuleConstant
from syft.service.response import SyftSuccess


@sy.api_endpoint_method()
def private_query_function(context, query_str: str) -> str:
    return query_str


@sy.api_endpoint_method()
def mock_query_function(context, query_str: str) -> str:
    return query_str


def test_constant(worker) -> None:
    root_client = worker.root_client
    new_endpoint = sy.TwinAPIEndpoint(
        path="test.test_query",
        description="Test",
        private_function=private_query_function,
        mock_function=mock_query_function,
    )

    res = root_client.api.services.api.add(endpoint=new_endpoint)

    assert isinstance(res, SyftSuccess)

    create_constant = CreatePolicyRuleConstant(val=2)
    constant = create_constant.to_policy_rule("test")

    assert constant.val == 2
    assert constant.klass == int

    create_constant = CreatePolicyRuleConstant(
        val=root_client.api.services.test.test_query
    )
    constant = create_constant.to_policy_rule("test_2")

    assert constant.val == root_client.api.services.api[0].action_object_id
    assert constant.klass == CustomEndpointActionObject


def test_mixed_policy(worker, ds_client) -> None:
    root_client = worker.root_client

    ao = ActionObject.from_obj(2)
    ao = ao.send(ds_client)

    @sy.syft_function(
        input_policy=sy.MixedInputPolicy(
            arg_1=sy.Constant(val=1),
            arg_2=ao.id,
            arg_3=int,
            client=ds_client,
        )
    )
    def test(arg_1: int, arg_2: int, arg_3: int):
        return arg_1 + arg_2 + arg_3

    ds_client.code.request_code_execution(test)
    root_client.requests[0].approve()

    ds_client.code.test(arg_2=ao, arg_3=2)
