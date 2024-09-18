# stdlib
import asyncio
import random

# third party
from helpers.api import create_endpoints_query
from helpers.api import create_endpoints_schema
from helpers.api import create_endpoints_submit_query
from helpers.api import set_endpoint_settings
from helpers.events import Event
from helpers.events import EventManager
from helpers.events import Scenario
from helpers.fixtures_sync import make_client
from helpers.fixtures_sync import make_guest_client
from helpers.fixtures_sync import make_user
from helpers.fixtures_sync import sync_clients
from helpers.users import set_settings_allow_guest_signup
from helpers.workers import add_external_registry
from helpers.workers import check_worker_pool_exists
from helpers.workers import create_prebuilt_worker_image
from helpers.workers import create_worker_pool
from helpers.workers import get_prebuilt_worker_image
import pytest
from unsync import unsync

random.seed(42069)


@unsync
async def guest_user_setup_flow(_, events, user):
    user_client = make_guest_client(url="http://localhost:8081")
    print(f"Logged in as guest user {user.email}")
    user_client.forgot_password(email=user.email)
    print(f"Requested password reset {user.email}")


@unsync
async def user_low_side_activity(_, events, user, after=None):
    # loop: guest user creation is allowed
    if after:
        await events.await_for(event_name=after)

    guest_user_setup_flow(user.email)

    # login_user

    # submit_code
    # request_approval

    # loop: wait for approval

    # execute code
    # get result

    # dump result in a file
    pass


@unsync
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


@unsync
async def admin_create_worker_pool(
    _,
    admin_client,
    worker_pool_name,
    worker_docker_tag,
    events,
):
    """
    Worker pool creation typically involves
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

    # TODO
    # scale_worker_pool(
    #     events,
    #     admin_client,
    #     worker_pool_name,
    #     event_name=Event.WORKER_POOL_SCALED,
    #     after=Event.WORKER_POOL_CREATED,
    # )


@unsync
async def mark_completed(events, register, after):
    if after:
        await events.await_for(event_name=after)

    events.register(register)


@unsync
async def admin_low_side_activity(_, events):
    """
    Typical admin activity on low-side server
    1. Login to low-side server
    2. Enable guest sign up
    3. Create a worker pool
    3. Start checking requests every 'n' seconds
    """

    worker_pool_name = "bigquery-pool"
    worker_docker_tag = "openmined/worker-bigquery:0.9.1"

    # login to low side
    admin_client = make_client(
        url="http://localhost:8081",
        email="info@openmined.org",
        password="changethis",
    )

    # enable guest sign up
    set_settings_allow_guest_signup(
        events,
        admin_client,
        True,
        Event.ALLOW_GUEST_SIGNUP_ENABLED,
    )

    # create worker pool on low side
    admin_create_worker_pool(
        _,
        admin_client,
        worker_pool_name,
        worker_docker_tag,
        events,
    )

    # start checking requests every 5s
    # triage_requests(
    #     events,
    #     admin_client,
    #     register=Event.ADMIN_APPROVED_REQUEST,
    #     sleep=5,
    # )

    mark_completed(
        events,
        register=Event.ADMIN_LOW_SIDE_WORKFLOW_COMPLETED,
        after=Event.WORKER_POOL_CREATED,
    )


@unsync
async def admin_create_api_endpoint(
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


@unsync
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
        admin_client_high,
        worker_pool_name,
        worker_docker_tag,
        events,
    )

    admin_create_api_endpoint(
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

    # start admin activity on high side
    admin_low_side_activity(request, events)

    # todo
    # admin_high_side_activity(request, events)

    # todo - only start syncing after the root user created other admin users
    # root_sync_activity(request, events, after=Event.USER_ADMIN_CREATED)

    # todo
    [
        user_low_side_activity(
            request,
            events,
            make_user(),
            after=Event.ALLOW_GUEST_SIGNUP_ENABLED,
        )
        for i in range(5)
    ]

    await events.await_scenario(scenario_name="test_level_0_k8s", timeout=30)
    assert events.scenario_completed("test_level_0_k8s")
