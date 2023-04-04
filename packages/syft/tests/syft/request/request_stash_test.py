# stdlib

# third party

# syft absolute
from syft.core.node.new.client import SyftClient
from syft.core.node.new.credentials import SyftVerifyKey
from syft.core.node.new.request_stash import RequestStash

# def add_mock_requests(
#     request_stash: RequestStash, worker: Worker, guest_domain_client
# ) -> None:
#     print(worker)
#     print()
#     print(guest_domain_client)
#     print()
#     print(guest_domain_client.credentials)

#     request = SubmitRequest(changes=[CODE_EXECUTE])
#     result = request_stash.set(request.to(Request, context=context))


def test_requeststash_get_all_for_verify_key_no_requests(
    request_stash: RequestStash,
    guest_domain_client: SyftClient,
) -> None:
    # test when there are no requests from a client

    verify_key: SyftVerifyKey = guest_domain_client.credentials.verify_key
    requests = request_stash.get_all_for_verify_key(verify_key=verify_key)
    assert requests.is_ok() is True
    assert len(requests.ok()) == 0


# def test_requeststash_get_all_for_verify_key_error


#  def test_requeststash_get_all_for_status

#  def test_requeststash_get_all_for_status_error
