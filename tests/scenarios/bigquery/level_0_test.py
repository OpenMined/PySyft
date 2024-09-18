# stdlib
import asyncio
import random

# third party
from helpers.events import Event
from helpers.events import EventManager
from helpers.events import Scenario
from helpers.fixtures_sync import make_client
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


async def user_low_side_activity():
    # loop: guest user creation is allowed
    # create_user

    # login_user

    # submit_code
    # request_approval

    # loop: wait for approval

    # execute code
    # get result

    # dump result in a file
    pass


@unsync
async def admin_sync_activity(_, events, after):
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
        print("admin_sync_activity: syncing high & low")
        sync_clients(admin_client_high, admin_client_low)


@unsync
async def admin_create_worker_pool(_, admin_client, events):
    """
    Worker pool creation typically involves
    - Register custom image
    - Launch worker pool
    - Scale worker pool

    """

    worker_pool_name = "bigquery-pool"
    worker_docker_tag = "openmined/worker-bigquery:0.9.1"

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
async def admin_low_side_activity(_, events):
    """
    Typical admin activity on low-side server
    1. Login to low-side server
    2. Enable guest sign up
    3. Start checking requests every 'n' seconds
    """

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
    admin_create_worker_pool(_, admin_client, events)

    # start checking requests every 5s
    # triage_requests(
    #     events,
    #     admin_client,
    #     register=Event.ADMIN_APPROVED_REQUEST,
    #     sleep=5,
    # )

    events.register(Event.ADMIN_LOW_SIDE_WORKFLOW_COMPLETED)


@unsync
async def admin_high_side_activity(_, events):
    # login
    admin_client = make_client(
        url="http://localhost:8080",
        email="info@openmined.org",
        password="changethis",
    )

    admin_create_worker_pool(_, admin_client, events)

    events.register(Event.ADMIN_HIGH_SIDE_WORKFLOW_COMPLETED)


@pytest.mark.asyncio
async def test_level_0_k8s(request):
    scenario = Scenario(
        name="test_level_0_k8s",
        events=[
            Event.ALLOW_GUEST_SIGNUP_ENABLED,
            Event.ADMIN_LOW_SIDE_WORKFLOW_COMPLETED,
            Event.ADMIN_HIGH_SIDE_WORKFLOW_COMPLETED,
        ],
    )

    events = EventManager()
    events.add_scenario(scenario)
    events.monitor()

    # start admin activity on high side
    admin_low_side_activity(request, events)

    # todo
    admin_high_side_activity(request, events)

    # todo - only start syncing after the root user created other admin users
    admin_sync_activity(request, events, after=Event.USER_ADMIN_CREATED)

    # todo
    # users = create_users()
    # [user_low_side_activity(user) for user in users]

    await events.await_scenario(scenario_name="test_level_0_k8s", timeout=30)
    assert events.scenario_completed("test_level_0_k8s")
