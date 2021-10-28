# third party
import pytest
import torch

# syft absolute
import syft as sy

NETWORK_PORT = 9081
DOMAIN1_PORT = 9082


@pytest.mark.integration
def test_domain1_via_network_proxy_client() -> None:
    network_client = sy.login(
        email="info@openmined.org", password="changethis", port=NETWORK_PORT
    )
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    x = torch.Tensor([1, 2, 3])
    x_ptr = x.send(domain_client, tags=["my_tensor"])

    domain_list = network_client.domains.all(pandas=False)
    assert len(domain_list) > 0

    proxy_client = network_client.domains[domain_client.address.target_id.id]

    assert proxy_client.address == domain_client.address
    assert proxy_client.name == domain_client.name
    assert proxy_client.routes[0] != domain_client.routes[0]

    y_ptr = proxy_client.store[x_ptr.id_at_location.no_dash]
    assert x_ptr.id_at_location == y_ptr.id_at_location
    assert type(x_ptr).__name__ == type(y_ptr).__name__
