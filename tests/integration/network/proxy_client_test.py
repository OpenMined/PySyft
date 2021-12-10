# stdlib
import time
import uuid

# third party
import pytest
from tests.integration.conftest import TestNodeData
import torch

# syft absolute
import syft as sy
from syft.core.node.common.action.exception_action import UnknownPrivateException


@pytest.mark.network
def test_domain1_via_network_proxy_client(test_network: TestNodeData, test_domain_1: TestNodeData) -> None:
    unique_tag = str(uuid.uuid4())
    network_client = sy.login(
        email="info@openmined.org", password="changethis", url=test_network.grid_api_url
    )

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", url=test_domain_1.grid_api_url
    )

    x = torch.Tensor([1, 2, 3])
    x_ptr = x.send(domain_client, tags=[unique_tag])

    time.sleep(1)

    _ = domain_client.store[x_ptr.id_at_location.no_dash]

    domain_list = network_client.domains.all(pandas=False)
    assert len(domain_list) > 0
    proxy_client = network_client.domains[domain_client.address.target_id.id]

    assert proxy_client.address == domain_client.address
    assert proxy_client.name == domain_client.name
    assert proxy_client.routes[0] != domain_client.routes[0]

    time.sleep(1)

    y_ptr = proxy_client.store[x_ptr.id_at_location.no_dash]
    assert x_ptr.id_at_location == y_ptr.id_at_location
    assert type(x_ptr).__name__ == type(y_ptr).__name__


@pytest.mark.network
def test_search_network(test_network: TestNodeData, test_domain_1: TestNodeData) -> None:
    unique_tag = str(uuid.uuid4())
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", url=test_domain_1.grid_api_url
    )

    x = torch.Tensor([1, 2, 3])
    x.send(domain_client, tags=[unique_tag])

    network_client = sy.login(url=test_network.grid_api_url)

    query = [unique_tag]
    result = network_client.search(query=query, pandas=False)

    assert len(result) == 1
    assert result[0]["name"] == "test_domain_1"
    assert result[0]["host_or_ip"] == test_domain_1.vpn_ip


@pytest.mark.network
def test_proxy_login_logout_network(test_network: TestNodeData, test_domain_1: TestNodeData) -> None:
    unique_tag = str(uuid.uuid4())

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", url=test_domain_1.grid_api_url
    )

    x = torch.Tensor([1, 2, 3])
    x.send(domain_client, tags=[unique_tag])

    network_client = sy.login(url=test_network.grid_api_url)
    domain_list = network_client.domains.all(pandas=False)
    assert len(domain_list) > 0

    proxy_client = network_client.domains[domain_client.id.no_dash]

    # cant get it as a guest
    with pytest.raises(UnknownPrivateException):
        proxy_client.store[unique_tag].get(delete_obj=False)

    proxy_client.login(email="info@openmined.org", password="changethis")
    res = proxy_client.store[unique_tag].get(delete_obj=False)
    assert (res == torch.Tensor([1, 2, 3])).all()

    proxy_client.logout()

    # cant get it as a guest
    with pytest.raises(UnknownPrivateException):
        proxy_client.store[unique_tag].get(delete_obj=False)
