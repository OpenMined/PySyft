# stdlib
import os
from textwrap import dedent

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.service.action.numpy import NumpyArrayObject
from syft.service.code.user_code import SubmitUserCode

LOCAL_ENCLAVE_PORT = os.environ.get("LOCAL_ENCLAVE_PORT", 8010)
# TODO: Should move to Docker Container tests


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


def test_manual_code_submission_enclave() -> None:
    # Step1: Login Phase
    canada_root = sy.Worker.named(name="canada", local_db=True, reset=True).root_client

    italy_root = sy.Worker.named(name="italy", local_db=True, reset=True).root_client

    # Step 2: Uploading to Domain Nodes
    load_dataset(canada_root)
    load_dataset(italy_root)

    assert sy.enable_external_lib("oblv")

    # Step 3: Connection to Enclave
    # TODO 🟣 Modify to use Data scientist account credentials
    # after Permission are integrated
    depl = sy.external.oblv.deployment_client.DeploymentClient(
        deployment_id="d-2dfedbb1-7904-493b-8793-1a9554badae7",
        oblv_client=None,
        domain_clients=[canada_root, italy_root],
        key_name="first",
    )  # connection_port key can be added to set the port on which oblv_proxy will run

    depl.initiate_connection(LOCAL_ENCLAVE_PORT)

    depl.register(
        name="Jane Doe",
        email="jane@caltech.edu",
        password="abc123",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )
    depl.login(email="jane@caltech.edu", password="abc123")

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
    print(depl.request_code_execution(code=simple_function))

    # Step 6: Code review phase
    canada_requests = canada_root.api.services.request.get_all()
    assert len(canada_requests) == 1
    assert canada_requests[0].approve()

    italy_requests = italy_root.api.services.request.get_all()
    assert len(italy_requests) == 1
    assert italy_requests[0].approve()

    assert hasattr(depl.api.services.code, "simple_function")
    res = depl.api.services.code.simple_function(
        canada_data=canada_data.assets[0], italy_data=italy_data.assets[0]
    )
    print(res, type(res))
    assert isinstance(res, NumpyArrayObject)
