# stdlib
import time
import uuid

# third party
import pytest
import torch

# syft absolute
import syft as sy
from syft.core.node.common.action.exception_action import UnknownPrivateException

NETWORK_PORT = 9081
DOMAIN1_PORT = 9082
DOMAIN1_VPN_IP = "100.64.0.2"


@pytest.mark.integration
def test_domain1_via_network_proxy_client() -> None:
    # stdlib
    from inspect import currentframe
    from inspect import getframeinfo

    print("running test_domain1_via_network_proxy_client")
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    unique_tag = str(uuid.uuid4())
    network_client = sy.login(
        email="info@openmined.org", password="changethis", port=NETWORK_PORT
    )
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    x = torch.Tensor([1, 2, 3])
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)
    x_ptr = x.send(domain_client, tags=[unique_tag])
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    time.sleep(1)
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    _ = domain_client.store[x_ptr.id_at_location.no_dash]
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    domain_list = network_client.domains.all(pandas=False)
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)
    assert len(domain_list) > 0
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    proxy_client = network_client.domains[domain_client.address.target_id.id]
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    assert proxy_client.address == domain_client.address
    assert proxy_client.name == domain_client.name
    assert proxy_client.routes[0] != domain_client.routes[0]
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    time.sleep(1)
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)

    y_ptr = proxy_client.store[x_ptr.id_at_location.no_dash]
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)
    assert x_ptr.id_at_location == y_ptr.id_at_location
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)
    assert type(x_ptr).__name__ == type(y_ptr).__name__
    print("test_end_to_end_smpc_adp_trade_demo", getframeinfo(currentframe()).lineno)


@pytest.mark.integration
def test_search_network() -> None:
    unique_tag = str(uuid.uuid4())
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )

    x = torch.Tensor([1, 2, 3])
    x.send(domain_client, tags=[unique_tag])

    network_client = sy.login(port=NETWORK_PORT)

    query = [unique_tag]
    result = network_client.search(query=query, pandas=False)

    assert len(result) == 1
    assert result[0]["name"] == "test_domain_1"
    assert result[0]["host_or_ip"] == DOMAIN1_VPN_IP


@pytest.mark.integration
def test_proxy_login_logout_network() -> None:
    # stdlib
    from inspect import currentframe
    from inspect import getframeinfo

    print("running test_domain1_via_network_proxy_client")
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)

    unique_tag = str(uuid.uuid4())
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
    domain_client = sy.login(
        email="info@openmined.org", password="changethis", port=DOMAIN1_PORT
    )
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)

    x = torch.Tensor([1, 2, 3])
    x.send(domain_client, tags=[unique_tag])
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)

    network_client = sy.login(port=NETWORK_PORT)
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)

    domain_list = network_client.domains.all(pandas=False)
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
    assert len(domain_list) > 0

    proxy_client = network_client.domains[domain_client.id.no_dash]
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)

    # cant get it as a guest
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
    with pytest.raises(UnknownPrivateException):
        print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
        proxy_client.store[unique_tag].get(delete_obj=False)
        print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)

    proxy_client.login(email="info@openmined.org", password="changethis")
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)

    res = proxy_client.store[unique_tag].get(delete_obj=False)
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
    assert (res == torch.Tensor([1, 2, 3])).all()

    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
    proxy_client.logout()
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
    # cant get it as a guest
    with pytest.raises(UnknownPrivateException):
        print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
        proxy_client.store[unique_tag].get(delete_obj=False)
    print("test_proxy_login_logout_network", getframeinfo(currentframe()).lineno)
