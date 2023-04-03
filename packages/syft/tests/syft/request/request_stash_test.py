# syft absolute
from syft.core.node.new.request_stash import RequestStash
from syft.core.node.new.request_stash import RequestingUserVerifyKeyPartitionKey


def test_requeststash_get_all_for_verify_key(request_stash: RequestStash):
    RequestingUserVerifyKeyPartitionKey.with_obj("test")
    request_stash.get_all_for_verify_key(verify_key="test")
    print(request_stash)

    print("Hello World")
    assert False


#  def test_requeststash_get_all_for_status
