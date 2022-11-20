# stdlib
from datetime import datetime
import time
import uuid

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.store.proxy_dataset import ProxyDataset
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE
from syft.util import size_mb

DOMAIN1_PORT = 9082

@pytest.mark.e2e
def test_budget() -> None:

    # use to enable mitm proxy
    # from syft.grid.connections.http_connection import HTTPConnection
    # HTTPConnection.proxies = {"http": "http://127.0.0.1:8080"}

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )
    ages = np.array([25, 35, 21, 19, 40, 55, 31, 18, 27, 33])
    names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


    phi_tensor = sy.Tensor(ages).private(min_val=0, max_val=122, data_subjects=names)

    domain_client.load_dataset(
        assets={"ages_tensor": phi_tensor},
        name="ages_dataset",
        description="Ages of a group of people"
    )

    starting_budget = 999999
    data_scientist_details = {
        "name": "Rey Skywalker",
        "email": "rey@skywalker.net",
        "password": "jakku",
        "budget": starting_budget,
    }
    domain_client.users.create(**data_scientist_details)
    
    skywalker = sy.login(port=DOMAIN1_PORT, email="rey@skywalker.net", password="jakku")
    
    dataset_prt = skywalker.datasets[-1]["ages_tensor"]
    mean_ptr = dataset_prt.mean()
    
    starting_budget = skywalker.privacy_budget
    
    result_ptr = mean_ptr.publish(sigma=1.5)
    result_ptr.block_with_timeout(60)
    _ = result_ptr.get(delete_obj=False)
    
    current_budget = skywalker.privacy_budget
    budget_spent = 308.9642958587137
    print(starting_budget- current_budget - budget_spent)
    assert starting_budget- current_budget == budget_spent

    
