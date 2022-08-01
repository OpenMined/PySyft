# stdlib
from concurrent import futures
import os
import time

# third party
import pytest

# syft absolute
import syft as sy
from syft import DomainClient
from syft.core.node.common.node_service.sleep.sleep_messages import (
    SleepMessageWithReply,
)
from syft.core.node.common.node_service.sleep.sleep_messages import SleepReplyMessage

NETWORK_PORT = 9081
HOST_IP = os.environ.get("HOST_IP", "localhost")
NETWORK_PUBLIC_HOST = f"{HOST_IP}:{NETWORK_PORT}"
print("Network IP", NETWORK_PUBLIC_HOST)
DOMAIN1_PORT = 9082
NETWORK_VPN_IP = "100.64.0.1"
DOMAIN1_VPN_IP = "100.64.0.2"
TEST_ROOT_EMAIL = "info@openmined.org"
TEST_ROOT_PASS = "changethis"

EMULATION = os.environ.get("EMULATION", "false")
print("EMULATION", EMULATION)


def send_msg(domain: DomainClient) -> SleepMessageWithReply:
    msg = SleepMessageWithReply(kwargs={"seconds": 0.5}).to(
        address=domain.address, reply_to=domain.address
    )
    return domain.send_immediate_msg_with_reply(msg=msg)


@pytest.mark.network
def test_parallel_sync_io_requests() -> None:
    domain = sy.login(port=DOMAIN1_PORT, email=TEST_ROOT_EMAIL, password=TEST_ROOT_PASS)

    # FastAPI and Starlette use anyio under the hood to use a threadpool for non async
    # CapacityLimiter in anyio is default 40 threads
    request_count = 40

    start = time.time()
    with futures.ThreadPoolExecutor(max_workers=request_count) as executor:
        res = list(executor.map(send_msg, [domain] * request_count))
    end = time.time()

    total = end - start

    assert len(res) == request_count
    for i in res:
        assert isinstance(i, SleepReplyMessage)

    expected_time = 60
    if EMULATION != "false":
        expected_time = 70  # emulation is slow on CI
    assert total <= expected_time
