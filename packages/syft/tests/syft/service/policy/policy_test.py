# third party
import pytest

# syft absolute
from syft import Asset
from syft import Constant
from syft import Dataset
from syft import MixedInputPolicy
from syft import syft_function
from syft.client.api import AuthedServiceContext
from syft.service.user.user_roles import ServiceRole
from syft.types.errors import SyftException


@pytest.fixture
def submit_code_with_constants_only(ds_client, worker):
    input_policy = MixedInputPolicy(
        endpoint=Constant(val="TEST ENDPOINT"),
        query=Constant(val="TEST QUERY"),
        client=ds_client,
    )

    @syft_function(
        input_policy=input_policy,
    )
    def test_func():
        return 1

    admin_client = worker.root_client

    ds_client.code.submit(test_func)

    user_code = admin_client.api.services.code[0]

    yield user_code


@pytest.fixture
def submit_code_with_mixed_inputs(ds_client, worker):
    admin_client = worker.root_client
    ds = Dataset(name="test", asset_list=[Asset(name="test", data=[1, 2], mock=[2, 3])])

    admin_client.upload_dataset(ds)

    asset = ds_client.datasets[0].assets[0]

    mix_input_policy = MixedInputPolicy(
        data=asset,
        endpoint=Constant(val="TEST ENDPOINT"),
        query=Constant(val="TEST QUERY"),
        client=ds_client,
    )

    @syft_function(
        input_policy=mix_input_policy,
    )
    def test_func_data(data, test_basic_python_type):
        return data

    admin_client = worker.root_client

    ds_client.code.submit(test_func_data)

    user_code = admin_client.api.services.code[0]

    yield user_code


class TestMixedInputPolicy:
    def test_constants_not_required(self, submit_code_with_constants_only):
        user_code = submit_code_with_constants_only

        policy = user_code.input_policy

        assert policy.is_valid(context=None, usr_input_kwargs={})

    def test_providing_constants_valid(self, submit_code_with_constants_only):
        user_code = submit_code_with_constants_only

        policy = user_code.input_policy

        assert policy.is_valid(
            context=None,
            usr_input_kwargs={"endpoint": "TEST ENDPOINT", "query": "TEST QUERY"},
        )

    def test_constant_vals_can_be_retrieved_by_admin(
        self, submit_code_with_constants_only
    ):
        user_code = submit_code_with_constants_only

        policy = user_code.input_policy

        mapped_inputs = {k: v.val for k, v in list(policy.inputs.values())[0].items()}

        assert mapped_inputs == {"endpoint": "TEST ENDPOINT", "query": "TEST QUERY"}

    def test_mixed_inputs_invalid_without_same_ds(self, submit_code_with_mixed_inputs):
        user_code = submit_code_with_mixed_inputs

        policy = user_code.input_policy

        with pytest.raises(SyftException):
            policy.is_valid(context=None, usr_input_kwargs={})

    def test_mixed_inputs_valid_with_same_asset(
        self, worker, ds_client, submit_code_with_mixed_inputs
    ):
        user_code = submit_code_with_mixed_inputs

        policy = user_code.input_policy

        asset = ds_client.datasets[0].assets[0]
        ds_context = AuthedServiceContext(
            server=worker,
            credentials=ds_client.verify_key,
            role=ServiceRole.DATA_SCIENTIST,
        )
        assert policy.is_valid(
            context=ds_context, usr_input_kwargs={"data": asset.action_id}
        )

    def test_mixed_inputs_invalid_with_different_asset_raises(
        self, worker, ds_client, submit_code_with_mixed_inputs
    ):
        admin_client = worker.root_client

        ds = Dataset(
            name="different ds",
            asset_list=[Asset(name="different asset", data=[1, 2], mock=[2, 3])],
        )
        admin_client.upload_dataset(ds)
        user_code = submit_code_with_mixed_inputs

        policy = user_code.input_policy

        asset = ds_client.datasets["different ds"].assets[0]
        ds_context = AuthedServiceContext(
            server=worker,
            credentials=ds_client.verify_key,
            role=ServiceRole.DATA_SCIENTIST,
        )
        with pytest.raises(SyftException):
            policy.is_valid(
                context=ds_context, usr_input_kwargs={"data": asset.action_id}
            )
