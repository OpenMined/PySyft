# third party
import numpy as np
import pytest

# syft absolute
import syft as sy

CANADA_DOMAIN_PORT = 9082
ITALY_DOMAIN_PORT = 9083
LOCAL_ENCLAVE_PORT = 8010


@pytest.mark.oblv
def test_dataset_upload_to_enclave() -> None:

    data_scientist = {
        "name": "DS",
        "email": "DS@om.com",
        "password": "enclave",
        "budget": 1000000,
    }

    ca_root = sy.login(
        email="info@openmined.org", password="changethis", port=CANADA_DOMAIN_PORT
    )
    it_root = sy.login(
        email="info@openmined.org", password="changethis", port=ITALY_DOMAIN_PORT
    )

    data1 = np.array([1, 2, 3, 4, 5])
    dataset1 = sy.Tensor(data1).private(min_val=0, max_val=5, data_subject="test_data1")
    data2 = np.array([5, 4, 3, 2, 1])
    dataset2 = sy.Tensor(data2).private(min_val=0, max_val=5, data_subject="test_data2")

    canada_ptr = dataset1.send(ca_root)
    italy_ptr = dataset2.send(it_root)

    ca_root.create_user(**data_scientist)
    it_root.create_user(**data_scientist)

    canada = sy.login(port=CANADA_DOMAIN_PORT, email="DS@om.com", password="enclave")
    italy = sy.login(port=ITALY_DOMAIN_PORT, email="DS@om.com", password="enclave")

    depl = sy.oblv.deployment_client.DeploymentClient(
        deployment_id="d-2dfedbb1-7904-493b-8793-1a9554badae7",
        oblv_client=None,
        domain_clients=[canada, italy],
        user_key_name="first",
    )  # connection_port key can be added to set the port on which oblv_proxy will run

    depl.initiate_connection(LOCAL_ENCLAVE_PORT)

    assert depl.get_uploaded_datasets() == []
    canada.oblv.transfer_dataset(
        deployment=depl, dataset=canada_ptr.id_at_location.no_dash
    )
    italy.oblv.transfer_dataset(
        deployment=depl, dataset=italy_ptr.id_at_location.no_dash
    )

    uploaded_datasets = [x["id"] for x in depl.get_uploaded_datasets()]
    assert len(uploaded_datasets) == 2
    assert canada_ptr.id_at_location.no_dash in uploaded_datasets
    assert italy_ptr.id_at_location.no_dash in uploaded_datasets
