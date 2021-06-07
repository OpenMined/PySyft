# third party
import pytest
from pytest import approx

# syft absolute
import syft as sy

psi = pytest.importorskip("openmined_psi")
sy.load("openmined_psi")


@pytest.mark.parametrize("reveal_intersection", [True, False])
@pytest.mark.vendor(lib="openmined_psi")
def test_psi(reveal_intersection: bool, node: sy.VirtualMachine) -> None:
    server_vm = node.get_root_client()
    client_vm = node.get_root_client()

    # server send reveal_intersection
    s_reveal_intersection = reveal_intersection
    s_sy_reveal_intersection = sy.lib.python.Bool(s_reveal_intersection)
    s_sy_reveal_intersection.send(
        server_vm,
        pointable=True,
        tags=["reveal_intersection"],
        description="reveal intersection value",
    )
    assert (
        server_vm.store["reveal_intersection"].description
        == "reveal intersection value"
    )

    # client get reval_intersection
    c_reveal_intersection = server_vm.store["reveal_intersection"].get()
    assert c_reveal_intersection == s_reveal_intersection

    # server send fpr
    s_fpr = 1e-6
    s_sy_fpr = sy.lib.python.Float(s_fpr)
    s_sy_fpr.send(
        server_vm, pointable=True, tags=["fpr"], description="false positive rate"
    )

    # client get fpr
    c_fpr = server_vm.store["fpr"].get()
    assert c_fpr == approx(s_fpr)

    # client send client_items_len
    psi_client = psi.client.CreateWithNewKey(c_reveal_intersection)
    c_items = ["Element " + str(i) for i in range(1000)]
    c_sy_client_items_len = sy.lib.python.Int(len(c_items))
    c_sy_client_items_len.send(
        client_vm,
        pointable=True,
        tags=["client_items_len"],
        description="client items length",
    )

    # server get client_items_len
    s_sy_client_items_len = client_vm.store["client_items_len"].get(delete_obj=False)
    assert s_sy_client_items_len == c_sy_client_items_len

    # server send setup message
    s_items = ["Element " + str(2 * i) for i in range(1000)]
    psi_server = psi.server.CreateWithNewKey(s_reveal_intersection)
    s_setup = psi_server.CreateSetupMessage(s_fpr, s_sy_client_items_len, s_items)
    s_setup.send(
        server_vm,
        pointable=True,
        tags=["setup"],
        description="psi.server Setup Message",
    )
    assert server_vm.store["setup"].description == "psi.server Setup Message"

    # client get setup message
    c_setup = server_vm.store["setup"].get()
    assert c_setup == s_setup

    # client send request
    c_request = psi_client.CreateRequest(c_items)
    c_request.send(
        client_vm, tags=["request"], pointable=True, description="client request"
    )

    # server get request
    s_request = client_vm.store["request"].get()
    assert s_request == c_request

    # server send response
    s_response = psi_server.ProcessRequest(s_request)
    s_response.send(
        server_vm, pointable=True, tags=["response"], description="psi.server response"
    )

    # client get response
    c_response = server_vm.store["response"].get()
    assert c_response == s_response

    # client get result
    if c_reveal_intersection:
        intersection = psi_client.GetIntersection(c_setup, c_response)
        iset = set(intersection)
        for idx in range(len(c_items)):
            if idx % 2 == 0:
                assert idx in iset
            else:
                assert idx not in iset
    else:
        intersection = psi_client.GetIntersectionSize(c_setup, c_response)
        assert intersection >= (len(c_items) / 2.0)
        assert intersection <= (1.1 * len(c_items) / 2.0)
