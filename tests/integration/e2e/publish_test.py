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


def highest() -> int:
    ii64 = np.iinfo(DEFAULT_INT_NUMPY_TYPE)
    return ii64.max

# TODO: this is only the basic flow, we need to identify a way to test
# flows that are happening on the worker container
@pytest.mark.e2e
def test_publish():
    
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )
    
    fred_nums = np.array([25, 35, 21, 19, 40, 55, 31, 18, 27, 33])
    velma_nums = np.array([8, 11, 10, 50, 44, 32, 55, 29, 6, 1])
    assert len(fred_nums) == len(velma_nums)
    fred_tensor = sy.Tensor(fred_nums).private(min_val=0, max_val=122, data_subjects="fred")
    velma_tensor = sy.Tensor(velma_nums).private(min_val=0, max_val=122, data_subjects="velma")
    
    domain_client.create_user(
        name="Scooby",
        email="scooby@doo.org",
        password="snacks",
        budget=0,
    )
    
    domain_client.load_dataset(
            assets={"fred_nums": fred_tensor},
            name="Fred's Data",
            description="This is Fred's private Data",
        )
    
    domain_client.load_dataset(
            assets={"velma_nums": velma_tensor},
            name="Velma's Data",
            description="This is Velma's private Data",
        )
    
    ds_client = sy.login(
        email="scooby@doo.org", password="snacks", port=DOMAIN1_PORT 
    )
    
    fred_nums_ptr = ds_client.datasets[0]["fred_nums"]
    velma_nums_ptr = ds_client.datasets[1]["velma_nums"]
    comb_ptr = fred_nums_ptr + velma_nums_ptr
    op1 = comb_ptr * 2
    op2 = op1.sum()
    res = op2.publish(sigma=1.5)
