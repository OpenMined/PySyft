# stdlib
from concurrent import futures
import os

# syft absolute
import syft as sy
from syft import DomainClient
from syft.core.node.common.node_service.object_search.obj_search_service import (
    ObjectSearchMessage,
)
from syft.core.node.common.node_service.object_search.obj_search_service import (
    ObjectSearchReplyMessage,
)

NETWORK_PORT = 9081
HOST_IP = os.environ.get("HOST_IP", "localhost")
NETWORK_PUBLIC_HOST = f"{HOST_IP}:{NETWORK_PORT}"
print("Network IP", NETWORK_PUBLIC_HOST)
DOMAIN1_PORT = 9082
NETWORK_VPN_IP = "100.64.0.1"
DOMAIN1_VPN_IP = "100.64.0.2"
TEST_ROOT_EMAIL = "info@openmined.org"
TEST_ROOT_PASS = "changethis"


def send_msg(domain: DomainClient) -> ObjectSearchReplyMessage:
    msg = ObjectSearchMessage(address=domain.address, reply_to=domain.address)
    return domain.send_immediate_msg_with_reply(msg=msg)


def test_parallel_sync_io_requests() -> None:
    domain = sy.login(port=DOMAIN1_PORT, email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS)

    with futures.ThreadPoolExecutor(max_workers=100) as executor:
        res = list(executor.map(send_msg, [domain] * 300))

    assert len(res) == 300
    for i in res:
        assert isinstance(i, ObjectSearchReplyMessage)
