# stdlib
import time
from typing import Any
from typing import Dict
from typing import Optional
import uuid

# third party
import pytest
import torch

# syft absolute
import syft as sy
from syft.core.node.common.action.exception_action import UnknownPrivateException
from syft.core.node.common.client import GET_OBJECT_TIMEOUT

NETWORK_PORT = 9081
DOMAIN1_PORT = 9082
DOMAIN2_PORT = 9083
DOMAIN1_VPN_IP = "100.64.0.2"


@pytest.mark.network
def test_domain1_via_network_proxy_client() -> None:
    unique_tag = str(uuid.uuid4())
    network_client = sy.login(
        email="info@openmined.org", password="changethis", port=NETWORK_PORT
    )

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    x = torch.Tensor([1, 2, 3])
    x_ptr = x.send(domain_client, tags=[unique_tag])

    time.sleep(5)

    _ = domain_client.store[x_ptr.id_at_location.no_dash]

    domain_list = network_client.domains.all(pandas=False)
    assert len(domain_list) > 0
    if sy.__version__ == "0.7.0":
        node_uid = domain_client.id
    else:
        node_uid = domain_client.node_uid

    proxy_client = network_client.domains[node_uid]

    if sy.__version__ == "0.7.0":
        proxy_client_uid = proxy_client.id
    else:
        proxy_client_uid = proxy_client.node_uid

    assert proxy_client_uid == node_uid
    assert proxy_client.name == domain_client.name
    assert proxy_client.routes[0] != domain_client.routes[0]

    time.sleep(1)

    retry_time = 5
    while retry_time > 0:
        retry_time -= 1
        try:
            y_ptr = proxy_client.store[x_ptr.id_at_location.no_dash]
            break
        except Exception as e:
            print(e)
            print("Retrying")
            time.sleep(1)

    assert x_ptr.id_at_location == y_ptr.id_at_location
    assert type(x_ptr).__name__ == type(y_ptr).__name__


@pytest.mark.network
def test_domain2_via_network_proxy_client() -> None:
    unique_tag = str(uuid.uuid4())
    network_client = sy.login(
        email="info@openmined.org", password="changethis", port=NETWORK_PORT
    )

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN2_PORT
    )

    x = torch.Tensor([1, 2, 3])
    x_ptr = x.send(domain_client, tags=[unique_tag])

    time.sleep(1)

    _ = domain_client.store[x_ptr.id_at_location.no_dash]

    domain_list = network_client.domains.all(pandas=False)
    assert len(domain_list) > 0

    if sy.__version__ == "0.7.0":
        node_uid = domain_client.id
    else:
        node_uid = domain_client.node_uid

    proxy_client = network_client.domains[node_uid]

    if sy.__version__ == "0.7.0":
        proxy_client_uid = proxy_client.id
    else:
        proxy_client_uid = proxy_client.node_uid

    proxy_client = network_client.domains[node_uid]

    assert proxy_client_uid == node_uid
    assert proxy_client.name == domain_client.name
    assert proxy_client.routes[0] != domain_client.routes[0]

    time.sleep(1)

    retry_time = 5
    while retry_time > 0:
        retry_time -= 1
        try:
            y_ptr = proxy_client.store[x_ptr.id_at_location.no_dash]
            break
        except Exception as e:
            print(e)
            print("Retrying")
            time.sleep(1)

    assert x_ptr.id_at_location == y_ptr.id_at_location
    assert type(x_ptr).__name__ == type(y_ptr).__name__


def search_network(network_client: sy.Network, query: str) -> Optional[Dict[str, Any]]:
    results = network_client.search(
        query=query, pandas=False, timeout=GET_OBJECT_TIMEOUT
    )

    for row in results:
        if row["host_or_ip"] == DOMAIN1_VPN_IP:
            return row
    return None


@pytest.mark.network
def test_search_network() -> None:
    unique_tag = str(uuid.uuid4())
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    x = torch.Tensor([1, 2, 3])
    x.send(domain_client, tags=[unique_tag])

    network_client = sy.login(port=NETWORK_PORT)
    query = [unique_tag]

    retry_time = 3
    while retry_time > 0:
        print(f"search_network attempt: {retry_time}")
        retry_time -= 1

        try:
            vpn_row = search_network(network_client=network_client, query=query)
            if vpn_row is not None:
                break
            else:
                time.sleep(10)
        except Exception as e:
            print(f"search_network failed. {e}")
            time.sleep(10)

    assert (
        vpn_row["name"].replace("-", "_") == "test_domain_1"
    )  # kubernetes forces - not _
    assert vpn_row["host_or_ip"] == DOMAIN1_VPN_IP


@pytest.mark.network
def test_proxy_login_logout_network() -> None:
    unique_tag = str(uuid.uuid4())

    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    x = torch.Tensor([1, 2, 3])
    x.send(domain_client, tags=[unique_tag])
    time.sleep(5)
    network_client = sy.login(port=NETWORK_PORT)
    domain_list = network_client.domains.all(pandas=False)
    assert len(domain_list) > 0

    proxy_client = network_client.domains[domain_client.id.no_dash]

    # cant get it as a guest
    with pytest.raises(UnknownPrivateException):
        proxy_client.store[unique_tag].get(delete_obj=False)

    proxy_client.login(email="info@openmined.org", password="changethis")
    store_obj = proxy_client.store[unique_tag]

    res = store_obj.get(delete_obj=False, timeout_secs=GET_OBJECT_TIMEOUT)
    assert (res == torch.Tensor([1, 2, 3])).all()

    proxy_client.logout()

    # cant get it as a guest
    with pytest.raises(UnknownPrivateException):
        proxy_client.store[unique_tag].get(delete_obj=False)
