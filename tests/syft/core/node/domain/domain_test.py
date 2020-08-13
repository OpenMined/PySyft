import torch as th
import syft as sy
from syft.core.node.domain import Domain, DomainClient
from syft import serialize, deserialize

def test_domain_creation():
    domain = Domain(name="test domain")

def test_domain_serde():
    domain_1 = Domain(name="domain 1")
    domain_1_client = domain_1.get_client()

    tensor = th.tensor([1, 2, 3])
    ptr = tensor.send(domain_1_client)

def test_domain_request_access():
    domain_1 = Domain(name="remote domain")
    tensor = th.tensor([1, 2, 3])
    domain_1_client = domain_1.get_client()
    data_ptr_domain_1 = tensor.send(domain_1_client)


    domain_2 = Domain(name='my domain"')
    domain_2_client = domain_2.get_client()

    data_ptr_domain_1.request_access(domain_2_client)

