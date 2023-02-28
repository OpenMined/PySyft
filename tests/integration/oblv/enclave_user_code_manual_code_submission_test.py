# stdlib
from textwrap import dedent

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.node.new.numpy import NumpyArrayObject
from syft.core.node.new.user_code import SubmitUserCode

PORT = 9082
CANADA_DOMAIN_PORT = PORT
ITALY_DOMAIN_PORT = PORT + 1
LOCAL_ENCLAVE_PORT = 8010


def load_dataset(domain_client) -> None:
    dataset_name = f"{domain_client.name}'s... Private Data"
    asset_name = "Secret data"
    dataset = sy.Dataset(name=dataset_name)
    asset = sy.Asset(name=asset_name)

    # Real Data
    x = np.array([1, 2, 3])
    asset.set_obj(x)

    # Mock Data
    y = np.array([1, 1, 1])
    asset.set_mock(y, mock_is_real=False)

    dataset.add_asset(asset)

    domain_client.upload_dataset(dataset)

    datasets = domain_client.datasets.get_all()

    assert len(datasets) == 1
    domain_dataset = datasets[0]
    assert domain_dataset.name == dataset_name
    assert len(domain_dataset.assets) == 1
    assert domain_dataset.assets[0].name == asset_name


@pytest.mark.oblv
def test_user_code_manual_code_submission_enclave() -> None:
    # Step1: Login Phase
    canada_root = sy.login(
        email="info@openmined.org", password="changethis", port=CANADA_DOMAIN_PORT
    )
    italy_root = sy.login(
        email="info@openmined.org", password="changethis", port=ITALY_DOMAIN_PORT
    )

    # Step 2: Uploading to Domain Nodes
    load_dataset(canada_root)
    load_dataset(italy_root)

    # Step 3: Connection to Enclave
    # TODO 🟣 Modify to use Data scientist account credentials
    # after Permission are integrated
    depl = sy.oblv.deployment_client.DeploymentClient(
        deployment_id="d-2dfedbb1-7904-493b-8793-1a9554badae7",
        oblv_client=None,
        domain_clients=[canada_root, italy_root],
        user_key_name="first",
    )  # connection_port key can be added to set the port on which oblv_proxy will run

    depl.initiate_connection(LOCAL_ENCLAVE_PORT)

    # Step 4: Manual code  preparation Phase
    canada_data = canada_root.datasets[-1]
    italy_data = italy_root.datasets[-1]

    @sy.syft_function(
        input_policy=sy.ExactMatch(
            canada_data=canada_data.assets[0], italy_data=italy_data.assets[0]
        ),
        output_policy=sy.SingleExecutionExactOutput(),
    )
    def simple_function(canada_data, italy_data):
        return canada_data + italy_data

    simple_function.code = dedent(simple_function.code)
    assert isinstance(simple_function, SubmitUserCode)

    # Step 5 :Code Submission Phase
    print(depl.new_request_code_execution(code=simple_function))

    # Step 6: Code review phase
    canada_requests = canada_root.api.services.request.get_all()
    assert len(canada_requests) == 1
    assert canada_requests[0].approve()

    italy_requests = italy_root.api.services.request.get_all()
    assert len(italy_requests) == 1
    assert italy_requests[0].approve()

    # Step 7: Result Retrieval Phase
    depl.refresh()

    assert hasattr(depl.api.services.code, "simple_function")
    res = depl.api.services.code.simple_function(
        canada_data=canada_data.assets[0], italy_data=italy_data.assets[0]
    )
    print(res)
    assert isinstance(res, NumpyArrayObject)
