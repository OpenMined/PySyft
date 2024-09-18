# stdlib
import uuid

# third party
from faker import Faker
import numpy as np
import pytest

# syft absolute
import syft as sy
from syft.client.datasite_client import DatasiteClient
from syft.server.worker import Worker
from syft.service.action.action_object import ActionObject
from syft.service.request.request import Request
from syft.service.request.request import UserCodeStatusChange
from syft.service.response import SyftError
from syft.service.response import SyftSuccess
from syft.service.user.user import User
from syft.service.user.user import UserView
from syft.service.user.user_roles import ServiceRole
from syft.types.errors import SyftException

# relative
from .user_test import ds_client as ds_client_fixture

ds_client = ds_client_fixture  # work around some ruff quirks


@sy.syft_function(
    input_policy=sy.ExactMatch(), output_policy=sy.SingleExecutionExactOutput()
)
def mock_syft_func():
    return 1


@sy.syft_function(
    input_policy=sy.ExactMatch(), output_policy=sy.SingleExecutionExactOutput()
)
def mock_syft_func_2():
    return 1


def test_repr_markdown_not_throwing_error(guest_client: DatasiteClient) -> None:
    guest_client.code.submit(mock_syft_func)
    result = guest_client.code.get_by_service_func_name("mock_syft_func")
    assert len(result) == 1
    assert result[0]._repr_markdown_()


def test_new_admin_can_list_user_code(
    worker: Worker,
    ds_client: DatasiteClient,
    faker: Faker,
) -> None:
    root_client = worker.root_client

    project = sy.Project(name="", members=[ds_client])
    project.create_code_request(mock_syft_func, ds_client)

    email = faker.email()
    pw = uuid.uuid4().hex
    root_client.register(
        name=faker.name(), email=email, password=pw, password_verify=pw
    )

    admin = root_client.login(email=email, password=pw)

    result: UserView = root_client.api.services.user.update(
        uid=admin.account.id, role=ServiceRole.ADMIN
    )
    assert result.role == ServiceRole.ADMIN

    user_code_stash = worker.services.user_code.stash
    user_codes = user_code_stash._data

    assert 1 == len(admin.code.get_all())
    assert {c.id for c in user_codes} == {c.id for c in admin.code}


def test_user_code(worker) -> None:
    root_datasite_client = worker.root_client
    root_datasite_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    guest_client = root_datasite_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    users = root_datasite_client.users.get_all()
    users[-1].allow_mock_execution()

    guest_client.api.services.code.request_code_execution(mock_syft_func)

    root_datasite_client = worker.root_client
    message = root_datasite_client.notifications[-1]
    request = message.link
    user_code = request.changes[0].code
    result = user_code.run()
    request.approve()

    result = guest_client.api.services.code.mock_syft_func()
    assert isinstance(result, ActionObject)

    real_result = result.get()
    assert isinstance(real_result, int)

    # Validate that the result is cached
    for _ in range(10):
        multi_call_res = guest_client.api.services.code.mock_syft_func()
        assert isinstance(result, ActionObject)
        assert multi_call_res.get() == result.get()


def test_duplicated_user_code(worker) -> None:
    worker.root_client.register(
        name="Jane Doe",
        email="jane@caltech.edu",
        password="abc123",
        password_verify="abc123",
        institution="Caltech",
        website="https://www.caltech.edu/",
    )
    ds_client = worker.root_client.login(
        email="jane@caltech.edu",
        password="abc123",
    )

    # mock_syft_func()
    result = ds_client.api.services.code.request_code_execution(mock_syft_func)
    assert isinstance(result, Request)
    assert len(ds_client.code.get_all()) == 1

    # request the exact same code should return an error
    with pytest.raises(SyftException):
        result = ds_client.api.services.code.request_code_execution(mock_syft_func)

    assert len(ds_client.code.get_all()) == 1

    # request the a different function name but same content will also succeed
    # flaky if not blocking
    mock_syft_func_2(syft_no_server=True)
    result = ds_client.api.services.code.request_code_execution(mock_syft_func_2)
    assert isinstance(result, Request)
    assert len(ds_client.code.get_all()) == 2

    code_history = ds_client.code_history
    assert code_history.code_versions, "No code version found."

    code_histories = worker.root_client.code_histories
    user_code_history = code_histories[ds_client.logged_in_user]
    assert not isinstance(code_histories, SyftError)
    assert not isinstance(user_code_history, SyftError)
    assert user_code_history.code_versions, "No code version found."
    assert user_code_history.mock_syft_func.user_code_history[0].status is not None
    assert user_code_history.mock_syft_func[0]._repr_markdown_(), "repr markdown failed"

    result = user_code_history.mock_syft_func_2[0]()
    assert result.get() == 1


def random_hash() -> str:
    return uuid.uuid4().hex[:16]


def test_scientist_can_list_code_assets(worker: sy.Worker, faker: Faker) -> None:
    asset_name = random_hash()
    asset = sy.Asset(
        name=asset_name, data=np.array([1, 2, 3]), mock=sy.ActionObject.empty()
    )
    dataset_name = random_hash()
    dataset = sy.Dataset(name=dataset_name, asset_list=[asset])

    root_client = worker.root_client

    password = random_hash()
    credentials = {
        "name": faker.name(),
        "email": faker.email(),
        "password": password,
        "password_verify": password,
    }

    root_client.register(**credentials)

    guest_client = root_client.guest()
    credentials.pop("name")
    guest_client = guest_client.login(**credentials)

    root_client.upload_dataset(dataset=dataset)

    asset_input = root_client.datasets.search(name=dataset_name)[0].asset_list[0]

    @sy.syft_function_single_use(asset=asset_input)
    def func(asset):
        return 0

    request = guest_client.code.request_code_execution(func)
    assert not isinstance(request, sy.SyftError)

    status_change = next(
        c for c in request.changes if (isinstance(c, UserCodeStatusChange))
    )

    assert status_change.code.assets[0].model_dump(
        mode="json"
    ) == asset_input.model_dump(mode="json")


@sy.syft_function()
def mock_inner_func():
    return 1


@sy.syft_function(
    input_policy=sy.ExactMatch(), output_policy=sy.SingleExecutionExactOutput()
)
def mock_outer_func(datasite):
    job = datasite.launch_job(mock_inner_func)
    return job


def test_nested_requests(worker, guest_client: User):
    guest_client.api.services.code.submit(mock_inner_func)
    guest_client.api.services.code.request_code_execution(mock_outer_func)

    root_datasite_client = worker.root_client
    request = root_datasite_client.requests[-1]

    root_datasite_client.api.services.request.apply(request.id)
    request = root_datasite_client.requests[-1]

    codes = root_datasite_client.code
    inner = codes[0] if codes[0].service_func_name == "mock_inner_func" else codes[1]
    outer = codes[0] if codes[0].service_func_name == "mock_outer_func" else codes[1]
    assert list(request.code.nested_codes.keys()) == ["mock_inner_func"]
    (linked_obj, server) = request.code.nested_codes["mock_inner_func"]
    assert server == {}
    resolved = root_datasite_client.api.services.notifications.resolve_object(
        linked_obj
    )
    assert resolved.id == inner.id
    assert outer.status.approved
    assert not inner.status.approved


def test_user_code_mock_execution(worker) -> None:
    # Setup
    root_datasite_client = worker.root_client

    # TODO guest_client fixture is not in root_datasite_client.users
    root_datasite_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client = root_datasite_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    dataset = sy.Dataset(
        name="my-dataset",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                data=np.array([0, 1, 2, 3, 4]),
                mock=np.array([5, 6, 7, 8, 9]),
            )
        ],
    )

    root_datasite_client.upload_dataset(dataset)

    # DS requests code execution
    data = ds_client.datasets[0].assets[0]

    @sy.syft_function_single_use(data=data)
    def compute_mean(data):
        return data.mean()

    ds_client.api.services.code.request_code_execution(compute_mean)

    # Guest attempts to set own permissions
    guest_user = ds_client.users.get_current_user()
    with pytest.raises(SyftException) as exc:
        guest_user.allow_mock_execution()

    assert "You are not permitted to perform this action." in exc.value.public_message

    # Mock execution fails, no permissions
    with pytest.raises(SyftException) as exc:
        result = ds_client.api.services.code.compute_mean(data=data.mock)

    assert (
        "You do not have the permissions for mock execution" in exc.value.public_message
    )

    # DO grants permissions
    users = root_datasite_client.users.get_all()
    guest_user = [u for u in users if u.id == guest_user.id][0]
    guest_user.allow_mock_execution()

    # Mock execution succeeds
    result = ds_client.api.services.code.compute_mean(data=data.mock).get()
    assert isinstance(result, float)


def test_mock_multiple_arguments(worker) -> None:
    # Setup
    root_datasite_client = worker.root_client

    root_datasite_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client = root_datasite_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    dataset = sy.Dataset(
        name="my-dataset",
        asset_list=[
            sy.Asset(
                name="numpy-data",
                data=np.array([0, 1, 2, 3, 4]),
                mock=np.array([5, 6, 7, 8, 9]),
            )
        ],
    )

    root_datasite_client.upload_dataset(dataset)
    users = root_datasite_client.users.get_all()
    users[-1].allow_mock_execution()

    # DS requests code execution
    data = ds_client.datasets[0].assets[0]

    @sy.syft_function_single_use(data1=data, data2=data)
    def compute_sum(data1, data2):
        return data1 + data2

    ds_client.api.services.code.request_code_execution(compute_sum)
    root_datasite_client.requests[-1].approve()

    # Mock execution succeeds, result not cached
    result = ds_client.api.services.code.compute_sum(data1=1, data2=1)
    assert result.get() == 2

    # Mixed execution fails on input policy
    with pytest.raises(SyftException) as exc:
        ds_client.api.services.code.compute_sum(data1=1, data2=data)

    assert exc.type is SyftException
    assert "not in allowed inputs" in exc.value.public_message

    # Real execution succeeds
    result = ds_client.api.services.code.compute_sum(data1=data, data2=data)

    # Mixed execution fails, no result from cache
    with pytest.raises(SyftException):
        result = ds_client.api.services.code.compute_sum(data1=1, data2=data)


def test_mock_no_arguments(worker) -> None:
    root_datasite_client = worker.root_client

    root_datasite_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client = root_datasite_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    users = root_datasite_client.users.get_all()

    @sy.syft_function_single_use()
    def compute_sum():
        return 1

    ds_client.api.services.code.request_code_execution(compute_sum)

    # not approved, no mock execution
    with pytest.raises(SyftException):
        ds_client.api.services.code.compute_sum()

    # not approved, mock execution
    users[-1].allow_mock_execution()
    result = ds_client.api.services.code.compute_sum()
    assert result, result
    assert result == 1

    # approved, no mock execution
    users[-1].allow_mock_execution(allow=False)
    message = root_datasite_client.notifications[-1]
    request = message.link
    user_code = request.changes[0].code
    result = user_code.run()
    request.approve()

    result = ds_client.api.services.code.compute_sum()
    # remove once we fix syft_action_saved_to_blob_store variable
    result = result.get()
    # uncomment once we fix syft_action_saved_to_blob_store
    # assert not isinstance(result.syft_action_data_cache, ActionDataEmpty)
    assert result == 1


def test_submit_invalid_name(worker) -> None:
    client = worker.root_client

    @sy.syft_function_single_use()
    def valid_name():
        pass

    res = client.code.submit(valid_name)
    assert isinstance(res, SyftSuccess)

    # reserved name
    with pytest.raises(SyftException):

        @sy.syft_function_single_use()
        def get_all():
            pass

    # no anonymous
    with pytest.raises(SyftException):

        @sy.syft_function_single_use()
        def _():
            pass

    # overwrite valid function name before submit, fail on serde
    @sy.syft_function_single_use()
    def valid_name_2():
        pass

    valid_name_2.func_name = "get_all"

    with pytest.raises(SyftException):
        client.code.submit(valid_name_2)


def test_submit_code_with_global_var(guest_client: DatasiteClient) -> None:
    with pytest.raises(SyftException) as exc:

        @sy.syft_function(
            input_policy=sy.ExactMatch(), output_policy=sy.SingleExecutionExactOutput()
        )
        def mock_syft_func_with_global():
            global x
            return x

    assert "Your code contains (a) global variable(s)" in exc.value.public_message

    with pytest.raises(SyftException) as exc:

        @sy.syft_function_single_use()
        def mock_syft_func_single_use_with_global():
            global x
            return x

    assert "Your code contains (a) global variable(s)" in exc.value.public_message


def test_request_existing_usercodesubmit(worker) -> None:
    root_datasite_client = worker.root_client

    root_datasite_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client = root_datasite_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    @sy.syft_function_single_use()
    def my_func():
        return 42

    res_submit = ds_client.api.services.code.submit(my_func)
    assert isinstance(res_submit, SyftSuccess)
    res_request = ds_client.api.services.code.request_code_execution(my_func)
    assert isinstance(res_request, Request)

    # Second request fails, cannot have multiple requests for the same code
    with pytest.raises(SyftException):
        res_request = ds_client.api.services.code.request_code_execution(my_func)

    assert len(ds_client.code.get_all()) == 1
    assert len(ds_client.requests.get_all()) == 1


def test_request_existing_usercode(worker) -> None:
    root_datasite_client = worker.root_client

    root_datasite_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client = root_datasite_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    @sy.syft_function_single_use()
    def my_func():
        return 42

    res_submit = ds_client.api.services.code.submit(my_func)
    assert isinstance(res_submit, SyftSuccess)

    code = ds_client.code.get_all()[0]
    res_request = ds_client.api.services.code.request_code_execution(my_func)
    assert isinstance(res_request, Request)

    # Second request fails, cannot have multiple requests for the same code
    with pytest.raises(SyftException):
        res_request = ds_client.api.services.code.request_code_execution(code)

    assert len(ds_client.code.get_all()) == 1
    assert len(ds_client.requests.get_all()) == 1


def test_submit_existing_code_different_user(worker):
    root_datasite_client = worker.root_client

    root_datasite_client.register(
        name="data-scientist",
        email="test_user@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client_1 = root_datasite_client.login(
        email="test_user@openmined.org",
        password="0000",
    )

    root_datasite_client.register(
        name="data-scientist-2",
        email="test_user_2@openmined.org",
        password="0000",
        password_verify="0000",
    )
    ds_client_2 = root_datasite_client.login(
        email="test_user_2@openmined.org",
        password="0000",
    )

    @sy.syft_function_single_use()
    def my_func():
        return 42

    res_submit = ds_client_1.api.services.code.submit(my_func)
    assert isinstance(res_submit, SyftSuccess)

    with pytest.raises(SyftException) as exc:
        ds_client_1.api.services.code.submit(my_func)

    assert "already exists" in exc.value.public_message

    # Resubmit with different user
    res_submit = ds_client_2.api.services.code.submit(my_func)
    assert isinstance(res_submit, SyftSuccess)

    with pytest.raises(SyftException) as exc:
        ds_client_2.api.services.code.submit(my_func)

    assert "already exists" in exc.value.public_message

    assert len(ds_client_1.code.get_all()) == 1
    assert len(ds_client_2.code.get_all()) == 1
    assert len(root_datasite_client.code.get_all()) == 2
