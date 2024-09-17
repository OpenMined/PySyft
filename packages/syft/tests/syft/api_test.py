# stdlib
from collections.abc import Callable

# third party
import numpy as np

# syft absolute
import syft as sy
from syft.service.response import SyftError
from syft.service.user.user_roles import ServiceRole


def test_api_cache_invalidation(worker):
    root_datasite_client = worker.root_client
    dataset = sy.Dataset(
        name="test",
        asset_list=[
            sy.Asset(
                name="test",
                data=np.array([1, 2, 3]),
                mock=np.array([1, 1, 1]),
                mock_is_real=False,
            )
        ],
    )
    root_datasite_client.upload_dataset(dataset)
    asset = root_datasite_client.datasets[0].assets[0]

    @sy.syft_function(
        input_policy=sy.ExactMatch(x=asset),
        output_policy=sy.SingleExecutionExactOutput(),
    )
    def my_func(x):
        return x + 1

    assert root_datasite_client.code.request_code_execution(my_func)
    # check that function is added to api without refreshing the api manually
    assert isinstance(root_datasite_client.code.my_func, Callable)


def test_api_cache_invalidation_login(root_verify_key, worker):
    guest_client = worker.guest_client
    worker.root_client.settings.allow_guest_signup(enable=True)
    assert guest_client.register(
        name="q", email="a@b.org", password="aaa", password_verify="aaa"
    )
    guest_client = guest_client.login(email="a@b.org", password="aaa")
    user_id = worker.root_client.users[-1].id

    def get_role(verify_key):
        users = worker.services.user.stash.get_all(root_verify_key).ok()
        user = [u for u in users if u.verify_key == verify_key][0]
        return user.role

    assert get_role(guest_client.credentials.verify_key) == ServiceRole.DATA_SCIENTIST

    dataset = sy.Dataset(
        name="test2",
    )
    assert isinstance(guest_client.upload_dataset(dataset), SyftError)

    assert guest_client.api.services.user.update(uid=user_id, name="abcdef")

    assert worker.root_client.api.services.user.update(
        uid=user_id, role=ServiceRole.DATA_OWNER
    )

    assert get_role(guest_client.credentials.verify_key) == ServiceRole.DATA_OWNER

    guest_client = guest_client.login(email="a@b.org", password="aaa")

    assert guest_client.upload_dataset(dataset)
