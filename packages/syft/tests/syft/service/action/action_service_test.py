# stdlib

# third party

# syft absolute
from syft.service.action.action_object import ActionObject
from syft.service.context import AuthedServiceContext

# TODO: Improve ActionService testing


def get_auth_ctx(worker):
    return AuthedServiceContext(
        server=worker, credentials=worker.signing_key.verify_key
    )


def test_action_service_sanity(worker):
    service = worker.services.action
    root_datasite_client = worker.root_client

    obj = ActionObject.from_obj("abc")
    pointer = obj.send(root_datasite_client)

    assert len(service.stash._data) == 1
    res = pointer.capitalize()
    assert res[0] == "A"
