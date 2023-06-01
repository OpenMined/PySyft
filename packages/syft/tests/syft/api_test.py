# stdlib
from textwrap import dedent
from typing import Callable

# third party
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.service.response import SyftAttributeError
from syft.service.user.user import UserUpdate
from syft.types.user_roles import ServiceRole


def test_api_cache_invalidation(worker):
    root_domain_client = worker.root_client
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
    root_domain_client.upload_dataset(dataset)
    asset = root_domain_client.datasets[0].assets[0]

    @sy.syft_function(
        input_policy=sy.ExactMatch(x=asset),
        output_policy=sy.SingleExecutionExactOutput(),
    )
    def my_func(x):
        return x + 1

    my_func.code = dedent(my_func.code)

    assert root_domain_client.api.services.code.request_code_execution(my_func)
    # check that function is added to api without refreshing the api manually
    assert isinstance(root_domain_client.api.services.code.my_func, Callable)


def test_api_cache_invalidation_login(root_verify_key, worker):
    guest_client = worker.guest_client
    assert guest_client.register(name="q", email="a@b.org", password="aaa")
    user_id = worker.document_store.partitions["User"].all(root_verify_key).value[-1].id

    def get_role(verify_key):
        users = worker.get_service("UserService").stash.get_all(root_verify_key).ok()
        user = [u for u in users if u.verify_key == verify_key][0]
        return user.role

    assert get_role(guest_client.credentials.verify_key) == ServiceRole.GUEST

    dataset = sy.Dataset(
        name="test2",
    )
    with pytest.raises(SyftAttributeError):
        assert guest_client.upload_dataset(dataset)

    assert guest_client.api.services.user.update(
        user_id, UserUpdate(user_id=user_id, name="abcdef")
    )

    assert worker.root_client.api.services.user.update(
        user_id, UserUpdate(user_id=user_id, role=ServiceRole.DATA_OWNER)
    )

    assert get_role(guest_client.credentials.verify_key) == ServiceRole.DATA_OWNER

    guest_client.login(email="a@b.org", password="aaa")

    assert guest_client.upload_dataset(dataset)
