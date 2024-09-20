# stdlib
import asyncio
from functools import wraps

# third party
from helpers.api import create_endpoints_query
from helpers.api import create_endpoints_schema
from helpers.api import create_endpoints_submit_query
from helpers.api import query_sql
from helpers.api import run_api_path
from helpers.api import set_endpoint_settings
from helpers.asserts import result_is
from helpers.code import run_code
from helpers.events import Event
from helpers.events import EventManager
from helpers.events import Scenario
from helpers.fixtures_sync import make_client
from helpers.fixtures_sync import make_guest_client
from helpers.fixtures_sync import make_user
from helpers.fixtures_sync import sync_clients
from helpers.workers import add_external_registry
from helpers.workers import check_worker_pool_exists
from helpers.workers import create_prebuilt_worker_image
from helpers.workers import create_worker_pool
from helpers.workers import get_prebuilt_worker_image
import pytest
from unsync import unsync

# syft absolute
import syft as sy


def unsync_guard():
    "Make sure we exit early if an exception occurs"

    def decorator(func):
        @wraps(func)
        @unsync
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"Exception occurred: {e}")
                for arg in args:
                    if isinstance(arg, EventManager):
                        print("Registering exception event")
                        arg.register(Event.EXCEPTION_OCCURRED)
                        break
                raise

        return wrapper

    return decorator


unsync_ = unsync_guard()
# unsync_ = unsync


@unsync_
async def guest_user_setup_flow(_, events, user):
    user_client = make_guest_client(url="http://localhost:8081")
    print(f"Logged in as guest user {user.email}")
    user_client.forgot_password(email=user.email)
    print(f"Requested password reset {user.email}")


@unsync_
async def user_low_side_activity(_, events, user, after=None):
    if after:
        await events.await_for(event_name=after)

    # login_user
    user_client = user.client()

    # submit_code
    submit_query_path = "bigquery.test_query"
    await result_is(
        events,
        lambda: len(run_api_path(user_client, submit_query_path, sql_query=query_sql()))
        == 10000,
        matches=True,
        after=[
            Event.QUERY_ENDPOINT_CONFIGURED,
            Event.USERS_CREATED_CHECKED,
            Event.ADMIN_SYNC_HIGH_TO_LOW,
        ],
        register=Event.USERS_CAN_QUERY_MOCK,
    )

    func_name = "test_func"
    await result_is(
        events,
        lambda: run_api_path(
            user_client,
            submit_query_path,
            func_name=func_name,
            query=query_sql(),
        ),
        matches="*Query submitted*",
        after=[Event.SUBMIT_QUERY_ENDPOINT_CONFIGURED, Event.USERS_CREATED_CHECKED],
        register=Event.USERS_CAN_SUBMIT_QUERY,
    )

    # this should fail to complete because no work will be approved or denied
    await result_is(
        events,
        lambda: run_code(user_client, method_name=f"{func_name}*"),
        matches=sy.SyftException(public_message="*Your code is waiting for approval*"),
        after=[Event.USERS_CAN_SUBMIT_QUERY],
        register=Event.USERS_QUERY_NOT_READY,
    )

    # dump result in a file

    events.register(Event.USER_LOW_SIDE_WAITING_FOR_APPROVAL)


@unsync_
async def root_sync_activity(_, events, after):
    if after:
        await events.await_for(event_name=after)

    # login to high side
    admin_client_high = make_client(
        url="http://localhost:8080",
        email="info@openmined.org",
        password="changethis",
    )

    admin_client_low = make_client(
        url="http://localhost:8081",
        email="info@openmined.org",
        password="changethis",
    )

    while True:
        await asyncio.sleep(3)
        sync_clients(
            events,
            admin_client_low,
            admin_client_high,
            event_name=Event.ADMIN_SYNC_LOW_TO_HIGH,
        )


@unsync_
async def admin_create_worker_pool(
    _,
    events,
    admin_client,
    worker_pool_name,
    worker_docker_tag,
):
    """
    Worker pool flow:
    - Register custom image
    - Launch worker pool
    - Scale worker pool
    """

    create_prebuilt_worker_image(
        events,
        admin_client,
        worker_docker_tag,
        Event.PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
    )

    worker_image_result = get_prebuilt_worker_image(
        events,
        admin_client,
        worker_docker_tag,
        after=Event.PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
    )

    # todo - configure this manually??
    add_external_registry(
        events,
        admin_client,
        Event.EXTERNAL_REGISTRY_BIGQUERY_CREATED,
    )

    create_worker_pool(
        events,
        admin_client,
        worker_pool_name,
        worker_image_result,
        Event.WORKER_POOL_CREATED,
    )

    check_worker_pool_exists(
        events,
        admin_client,
        worker_pool_name,
        after=Event.WORKER_POOL_CREATED,
    )


@unsync_
async def mark_completed(events, register, after):
    if after:
        await events.await_for(event_name=after)
    events.register(register)


@unsync_
async def admin_signup_users(_, events, admin_client, users, register):
    for user in users:
        print(f"Registering user {user.name} ({user.email})")
        admin_client.register(
            name=user.name,
            email=user.email,
            password=user.password,
            password_verify=user.password,
        )

    events.register(register)


@unsync_
async def admin_low_side_activity(_, events, users):
    """
    Typical admin activity on low-side server
    - Login to low-side server
    - Create users
    - Create a worker pool
    """

    worker_pool_name = "bigquery-pool"
    worker_docker_tag = "openmined/worker-bigquery:0.9.1"

    # login to low side
    admin_client = make_client(
        url="http://localhost:8081",
        email="info@openmined.org",
        password="changethis",
    )

    admin_signup_users(
        _,
        events,
        admin_client,
        users,
        register=Event.USERS_CREATED,
    )

    # create worker pool on low side
    admin_create_worker_pool(
        _,
        events,
        admin_client,
        worker_pool_name,
        worker_docker_tag,
    )

    mark_completed(
        events,
        register=Event.ADMIN_LOW_SIDE_WORKFLOW_COMPLETED,
        after=Event.WORKER_POOL_CREATED,
    )


@unsync_
async def admin_create_sync_api_endpoints(
    _,
    events,
    admin_client_high,
    admin_client_low,
    worker_pool_name,
    after=None,
):
    if after:
        await events.await_for(event_name=after)

    test_query_path = "bigquery.test_query"
    submit_query_path = "bigquery.submit_query"

    create_endpoints_query(
        events,
        admin_client_high,
        worker_pool_name=worker_pool_name,
        register=Event.QUERY_ENDPOINT_CREATED,
    )

    set_endpoint_settings(
        events,
        admin_client_high,
        path=test_query_path,
        kwargs={"endpoint_timeout": 120, "hide_mock_definition": True},
        after=Event.QUERY_ENDPOINT_CREATED,
        register=Event.QUERY_ENDPOINT_CONFIGURED,
    )

    create_endpoints_schema(
        events,
        admin_client_high,
        worker_pool_name=worker_pool_name,
        register=Event.SCHEMA_ENDPOINT_CREATED,
    )

    create_endpoints_submit_query(
        events,
        admin_client_high,
        worker_pool_name=worker_pool_name,
        register=Event.SUBMIT_QUERY_ENDPOINT_CREATED,
    )

    set_endpoint_settings(
        events,
        admin_client_high,
        path=submit_query_path,
        kwargs={"hide_mock_definition": True},
        after=Event.SUBMIT_QUERY_ENDPOINT_CREATED,
        register=Event.SUBMIT_QUERY_ENDPOINT_CONFIGURED,
    )

    sync_clients(
        events,
        admin_client_low,
        admin_client_high,
        event_name=Event.ADMIN_SYNC_HIGH_TO_LOW,
        after=Event.SUBMIT_QUERY_ENDPOINT_CONFIGURED,
    )


@unsync_
async def admin_high_side_activity(_, events):
    # login
    admin_client_high = make_client(
        url="http://localhost:8080",
        email="info@openmined.org",
        password="changethis",
    )
    admin_client_low = make_client(
        url="http://localhost:8081",
        email="info@openmined.org",
        password="changethis",
    )

    worker_pool_name = "bigquery-pool"
    worker_docker_tag = "openmined/worker-bigquery:0.9.1"

    admin_create_worker_pool(
        _,
        events,
        admin_client_high,
        worker_pool_name,
        worker_docker_tag,
    )

    admin_create_sync_api_endpoints(
        _,
        events,
        admin_client_high,
        admin_client_low,
        worker_pool_name,
        after=None,
    )

    events.register(Event.ADMIN_HIGH_SIDE_WORKFLOW_COMPLETED)


@pytest.mark.asyncio
async def test_level_0_k8s(request):
    """
    Goal
        - Setup two datasites - high & low
        - Root client of each datasite creates an multiple admin users

    """
    scenario = Scenario(
        name="test_level_0_k8s",
        events=[
            Event.ALLOW_GUEST_SIGNUP_ENABLED,
            Event.ADMIN_LOW_SIDE_WORKFLOW_COMPLETED,
            # Event.ADMIN_HIGH_SIDE_WORKFLOW_COMPLETED,
        ],
    )

    events = EventManager()
    events.add_scenario(scenario)
    events.monitor()

    users = [make_user(password="password") for _ in range(2)]

    # start admin activity on low side
    admin_low_side_activity(request, events, users)

    # todo
    admin_high_side_activity(request, events)

    # todo - only start syncing after the root user created other admin users
    # root_sync_activity(request, events, after=Event.USER_ADMIN_CREATED)

    # todo
    [
        user_low_side_activity(request, events, user, after=Event.USERS_CREATED)
        for user in users
    ]
    await events.await_scenario(scenario_name="test_level_0_k8s", timeout=30)
    assert events.scenario_completed("test_level_0_k8s")
