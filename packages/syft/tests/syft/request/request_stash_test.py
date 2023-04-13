# stdlib

# stdlib

# third party
import pytest
from pytest import MonkeyPatch
from result import Err

# syft absolute
from syft.core.node.new.client import SyftClient
from syft.core.node.new.context import AuthedServiceContext
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.document_store import QueryKeys
from syft.core.node.new.request import Request
from syft.core.node.new.request import RequestStatus
from syft.core.node.new.request import SubmitRequest
from syft.core.node.new.request_stash import RequestStash
from syft.core.node.new.request_stash import RequestingUserVerifyKeyPartitionKey
from syft.core.node.new.request_stash import StatusPartitionKey


def test_requeststash_get_all_for_verify_key_no_requests(
    request_stash: RequestStash,
    guest_domain_client: SyftClient,
) -> None:
    # test when there are no requests from a client

    verify_key: SyftVerifyKey = guest_domain_client.credentials.verify_key
    requests = request_stash.get_all_for_verify_key(verify_key=verify_key)
    assert requests.is_ok() is True
    assert len(requests.ok()) == 0


@pytest.mark.flaky(reruns=3, reruns_delay=1)
def test_requeststash_get_all_for_verify_key_success(
    request_stash: RequestStash,
    guest_domain_client: SyftClient,
    authed_context_guest_domain_client: AuthedServiceContext,
) -> None:
    # test when there is one request
    submit_request: SubmitRequest = SubmitRequest(changes=[])
    stash_set_result = request_stash.set(
        submit_request.to(Request, context=authed_context_guest_domain_client)
    )

    verify_key: SyftVerifyKey = guest_domain_client.credentials.verify_key
    requests = request_stash.get_all_for_verify_key(verify_key)

    assert requests.is_ok() is True
    assert len(requests.ok()) == 1
    assert requests.ok()[0] == stash_set_result.ok()

    # add another request
    submit_request_2: SubmitRequest = SubmitRequest(changes=[])
    stash_set_result_2 = request_stash.set(
        submit_request_2.to(Request, context=authed_context_guest_domain_client)
    )

    requests = request_stash.get_all_for_verify_key(verify_key)

    assert requests.is_ok() is True
    assert len(requests.ok()) == 2

    # the order might change so we check all requests
    assert (
        requests.ok()[1] == stash_set_result_2.ok()
        or requests.ok()[0] == stash_set_result_2.ok()
    )


def test_requeststash_get_all_for_verify_key_fail(
    request_stash: RequestStash,
    monkeypatch: MonkeyPatch,
    guest_domain_client: SyftClient,
) -> None:
    verify_key: SyftVerifyKey = guest_domain_client.credentials.verify_key
    mock_error_message = (
        "verify key not in the document store's unique or searchable keys"
    )

    def mock_query_all_error(qks: QueryKeys) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(request_stash, "query_all", mock_query_all_error)

    requests = request_stash.get_all_for_verify_key(verify_key)

    assert requests.is_err() is True
    assert requests.err() == mock_error_message


def test_requeststash_get_all_for_verify_key_find_index_fail(
    request_stash: RequestStash,
    monkeypatch: MonkeyPatch,
    guest_domain_client: SyftClient,
) -> None:
    verify_key: SyftVerifyKey = guest_domain_client.credentials.verify_key
    qks = QueryKeys(qks=[RequestingUserVerifyKeyPartitionKey.with_obj(verify_key)])

    mock_error_message = f"Failed to query index or search with {qks.all[0]}"

    def mock_find_index_or_search_keys_error(
        index_qks: QueryKeys, search_qks: QueryKeys
    ) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(
        request_stash.partition,
        "find_index_or_search_keys",
        mock_find_index_or_search_keys_error,
    )

    requests = request_stash.get_all_for_verify_key(verify_key)

    assert requests.is_err() is True
    assert requests.err() == mock_error_message


def test_requeststash_get_all_for_status_pending(
    request_stash: RequestStash,
    authed_context_guest_domain_client: AuthedServiceContext,
) -> None:
    submit_request = SubmitRequest(changes=[])
    request_stash.set(
        submit_request.to(Request, context=authed_context_guest_domain_client)
    )
    pending_status = RequestStatus.PENDING
    queried_requests = request_stash.get_all_for_status(pending_status)
    queried_requests = queried_requests.ok()

    assert len(queried_requests) == 1
    assert type(queried_requests[0]) == Request
    assert queried_requests[0].status == pending_status


def test_requeststash_get_all_for_status_approved(
    request_stash: RequestStash,
    authed_context_guest_domain_client: AuthedServiceContext,
) -> None:
    # test when there is one request with APPROVED status.
    # this is similar to when the status is REJECTED.
    submit_request = SubmitRequest(changes=[])
    request = submit_request.to(Request, context=authed_context_guest_domain_client)
    status = RequestStatus.APPROVED  # can also be REJECTED
    request.status = status
    stash_set_result = request_stash.set(request)
    assert stash_set_result.ok().status == status

    queried_requests = request_stash.get_all_for_status(status)
    queried_requests = queried_requests.ok()

    assert len(queried_requests) == 1
    assert type(queried_requests[0]) == Request
    assert queried_requests[0].status == status


def test_requeststash_get_all_for_status_fail(
    request_stash: RequestStash,
    monkeypatch: MonkeyPatch,
) -> None:
    pending_status = RequestStatus.PENDING
    qks = QueryKeys(qks=[StatusPartitionKey.with_obj(pending_status)])
    mock_error_message = f"Failed to query index or search with {qks.all[0]}"

    def mock_find_index_or_search_keys_error(
        index_qks: QueryKeys, search_qks: QueryKeys
    ) -> Err:
        return Err(mock_error_message)

    monkeypatch.setattr(
        request_stash.partition,
        "find_index_or_search_keys",
        mock_find_index_or_search_keys_error,
    )

    requests = request_stash.get_all_for_status(pending_status)

    assert requests.is_err() is True
    assert requests.err() == mock_error_message
