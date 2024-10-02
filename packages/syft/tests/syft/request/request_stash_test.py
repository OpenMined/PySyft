# third party

# syft absolute
from syft.client.client import SyftClient
from syft.server.credentials import SyftVerifyKey
from syft.service.context import AuthedServiceContext
from syft.service.request.request import Request
from syft.service.request.request import SubmitRequest
from syft.service.request.request_stash import RequestStash


def test_requeststash_get_all_for_verify_key_no_requests(
    root_verify_key,
    request_stash: RequestStash,
    guest_datasite_client: SyftClient,
) -> None:
    # test when there are no requests from a client

    verify_key: SyftVerifyKey = guest_datasite_client.credentials.verify_key
    requests = request_stash.get_all_for_verify_key(
        root_verify_key, verify_key=verify_key
    )
    assert requests.is_ok() is True
    assert len(requests.ok()) == 0


def test_requeststash_get_all_for_verify_key_success(
    root_verify_key,
    request_stash: RequestStash,
    guest_datasite_client: SyftClient,
    authed_context_guest_datasite_client: AuthedServiceContext,
) -> None:
    # test when there is one request
    submit_request: SubmitRequest = SubmitRequest(changes=[])
    stash_set_result = request_stash.set(
        root_verify_key,
        submit_request.to(Request, context=authed_context_guest_datasite_client),
    )

    verify_key: SyftVerifyKey = guest_datasite_client.credentials.verify_key
    requests = request_stash.get_all_for_verify_key(
        credentials=root_verify_key,
        verify_key=verify_key,
    )

    assert requests.is_ok() is True
    assert len(requests.ok()) == 1
    assert requests.ok()[0] == stash_set_result.ok()

    # add another request
    submit_request_2: SubmitRequest = SubmitRequest(changes=[])
    stash_set_result_2 = request_stash.set(
        root_verify_key,
        submit_request_2.to(Request, context=authed_context_guest_datasite_client),
    )

    requests = request_stash.get_all_for_verify_key(
        credentials=root_verify_key,
        verify_key=verify_key,
    )

    assert requests.is_ok() is True
    assert len(requests.ok()) == 2

    # the order might change so we check all requests
    assert (
        requests.ok()[1] == stash_set_result_2.ok()
        or requests.ok()[0] == stash_set_result_2.ok()
    )
