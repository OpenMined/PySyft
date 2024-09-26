# stdlib
import asyncio
from enum import auto
import random

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.service.request.request import RequestStatus

# relative
from .flows.admin_bigquery_api import bq_schema_endpoint
from .flows.admin_bigquery_api import bq_submit_endpoint
from .flows.admin_bigquery_api import bq_test_endpoint
from .flows.admin_bigquery_pool import bq_create_pool
from .flows.admin_common import register_user
from .flows.user_bigquery_api import bq_check_query_results
from .flows.user_bigquery_api import bq_submit_query
from .flows.user_bigquery_api import bq_test_query
from .sim.core import BaseEvent
from .sim.core import Simulator
from .sim.core import SimulatorContext
from .sim.core import sim_activity
from .sim.core import sim_entrypoint

fake = Faker()
NUM_USERS = 3
NUM_ENDPOINTS = 3  # test_query, submit_query, schema_query


class Event(BaseEvent):
    # overall state
    INIT = auto()
    ADMIN_LOWSIDE_FLOW_COMPLETED = auto()
    ADMIN_HIGHSIDE_FLOW_COMPLETED = auto()
    ADMIN_LOW_ALL_RESULTS_AVAILABLE = auto()
    USER_FLOW_COMPLETED = auto()
    # admin - endpoints
    ADMIN_ALL_ENDPOINTS_CREATED = auto()
    ADMIN_BQ_TEST_ENDPOINT_CREATED = auto()
    ADMIN_BQ_SUBMIT_ENDPOINT_CREATED = auto()
    ADMIN_BQ_SCHEMA_ENDPOINT_CREATED = auto()
    ADMIN_LOW_SIDE_ENDPOINTS_AVAILABLE = auto()
    # admin - worker pool
    ADMIN_WORKER_POOL_CREATED = auto()
    ADMIN_LOWSIDE_WORKER_POOL_CREATED = auto()
    ADMIN_HIGHSIDE_WORKER_POOL_CREATED = auto()
    # admin sync
    ADMIN_SYNC_COMPLETED = auto()
    ADMIN_SYNCED_HIGH_TO_LOW = auto()
    ADMIN_SYNCED_LOW_TO_HIGH = auto()
    # users
    GUEST_USERS_CREATED = auto()
    USER_CAN_QUERY_TEST_ENDPOINT = auto()
    USER_CAN_SUBMIT_QUERY = auto()
    USER_CHECKED_RESULTS = auto()


# ------------------------------------------------------------------------------------------------


@sim_activity(
    wait_for=Event.ADMIN_LOW_SIDE_ENDPOINTS_AVAILABLE,
    trigger=Event.USER_CAN_QUERY_TEST_ENDPOINT,
)
async def user_bq_test_query(ctx: SimulatorContext, client: sy.DatasiteClient):
    """Run query on test endpoint"""
    await asyncio.to_thread(bq_test_query, ctx, client)


@sim_activity(
    wait_for=Event.ADMIN_LOW_SIDE_ENDPOINTS_AVAILABLE,
    trigger=Event.USER_CAN_SUBMIT_QUERY,
)
async def user_bq_submit_query(ctx: SimulatorContext, client: sy.DatasiteClient):
    """Submit query to be run on private data"""
    await asyncio.to_thread(bq_submit_query, ctx, client)


@sim_activity(
    wait_for=Event.ADMIN_LOW_ALL_RESULTS_AVAILABLE,
    trigger=Event.USER_CHECKED_RESULTS,
)
async def user_bq_results(ctx: SimulatorContext, client: sy.DatasiteClient):
    await asyncio.to_thread(bq_check_query_results, ctx, client)


@sim_activity(wait_for=Event.GUEST_USERS_CREATED, trigger=Event.USER_FLOW_COMPLETED)
async def user_flow(ctx: SimulatorContext, server_url_low: str, user: dict):
    """
    User flow on low-side:
    - User logs in
    - User invokes the test query endpoint to get mock results - user_bq_test_query
    - User submits a query to be run on the private data for approval - user_bq_submit_query
    - User checks if request is approved and retrieves the results - user_bq_results

    The test -> submit -> results are typically done in sequence.
    test & submit can be done in parallel but results can be checked only after submit is done.
    """

    client = sy.login(
        url=server_url_low,
        email=user["email"],
        password=user["password"],
    )
    ctx.logger.info(f"User: {client.logged_in_user} - logged in")

    await user_bq_test_query(ctx, client)
    await user_bq_submit_query(ctx, client)
    await user_bq_results(ctx, client)


# ------------------------------------------------------------------------------------------------


@sim_activity(trigger=Event.GUEST_USERS_CREATED)
async def admin_register_users(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient, users: list[dict]
):
    await asyncio.gather(
        *[asyncio.to_thread(register_user, ctx, admin_client, user) for user in users],
    )


@sim_activity(trigger=Event.ADMIN_BQ_SCHEMA_ENDPOINT_CREATED)
async def admin_create_bq_schema_endpoint(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient, worker_pool: str
):
    await asyncio.to_thread(bq_schema_endpoint, ctx, admin_client, worker_pool)


@sim_activity(trigger=Event.ADMIN_BQ_TEST_ENDPOINT_CREATED)
async def admin_create_bq_test_endpoint(
    ctx: SimulatorContext,
    admin_client: sy.DatasiteClient,
    worker_pool: str,
):
    await asyncio.to_thread(bq_test_endpoint, ctx, admin_client, worker_pool)


@sim_activity(trigger=Event.ADMIN_BQ_SUBMIT_ENDPOINT_CREATED)
async def admin_create_bq_submit_endpoint(
    ctx: SimulatorContext,
    admin_client: sy.DatasiteClient,
    worker_pool: str,
):
    await asyncio.to_thread(bq_submit_endpoint, ctx, admin_client, worker_pool)


@sim_activity(trigger=Event.ADMIN_ALL_ENDPOINTS_CREATED)
async def admin_create_endpoints(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    worker_pool = "biquery-pool"

    await asyncio.gather(
        admin_create_bq_test_endpoint(ctx, admin_client, worker_pool),
        admin_create_bq_submit_endpoint(ctx, admin_client, worker_pool),
        admin_create_bq_schema_endpoint(ctx, admin_client, worker_pool),
    )
    ctx.logger.info("Admin high: Created all endpoints")


@sim_activity(
    wait_for=[
        Event.ADMIN_SYNCED_HIGH_TO_LOW,
        # endpoints work only after low side worker pool is created
        Event.ADMIN_LOWSIDE_WORKER_POOL_CREATED,
    ]
)
async def admin_watch_sync(ctx: SimulatorContext, admin_client: sy.DatasiteClient):
    while True:
        await asyncio.sleep(random.uniform(5, 10))

        # Check if endpoints are available
        endpoints = admin_client.custom_api.get_all()
        if len(endpoints) == NUM_ENDPOINTS:
            ctx.logger.info(
                f"Admin low: All {NUM_ENDPOINTS} API endpoints are synced from high."
            )
            ctx.logger.info(f"Endpoints: {endpoints}")
            ctx.events.trigger(Event.ADMIN_LOW_SIDE_ENDPOINTS_AVAILABLE)

        # Check if all requests are approved or denied
        requests = admin_client.requests.get_all()
        ctx.logger.info(f"Number of requests: {len(requests)}")
        if len(requests) == NUM_USERS:  # NOTE: currently hard coding this since
            # each user in `user_flow` submits 1 query request
            pending_requests = []
            for req in admin_client.requests:
                if req.get_status() == RequestStatus.PENDING:
                    pending_requests.append(req)
            if len(pending_requests) == 0:
                ctx.logger.info("Admin low: All requests are approved / denined.")
                ctx.logger.info(f"Requests: {requests}")
                ctx.events.trigger(Event.ADMIN_LOW_ALL_RESULTS_AVAILABLE)
                break
            else:
                ctx.logger.info(f"Admin low: Pending requests: {pending_requests}")


@sim_activity(trigger=Event.ADMIN_HIGHSIDE_WORKER_POOL_CREATED)
async def admin_create_bq_pool_high(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    await asyncio.to_thread(bq_create_pool, ctx, admin_client)


@sim_activity(trigger=Event.ADMIN_LOWSIDE_WORKER_POOL_CREATED)
async def admin_create_bq_pool_low(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    await asyncio.to_thread(bq_create_pool, ctx, admin_client)


@sim_activity(
    wait_for=[
        Event.USER_CAN_SUBMIT_QUERY,
        Event.ADMIN_SYNCED_LOW_TO_HIGH,
    ],
    trigger=Event.ADMIN_HIGHSIDE_FLOW_COMPLETED,
)
async def admin_triage_requests_high(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    while True:
        await asyncio.sleep(random.uniform(5, 10))

        # check if there are any requests
        # BUG: request that are executed request.code() are always in pending state
        pending_requests = [
            req
            for req in admin_client.requests
            if req.get_status() == RequestStatus.PENDING
        ]
        ctx.logger.info(f"Admin high: Found {len(pending_requests)} pending requests")
        for request in pending_requests:
            ctx.logger.info(f"Admin high: Found request {request.__dict__}")
            if getattr(request, "code", None):
                if "invalid_func" in request.code.service_func_name:
                    ctx.logger.info(f"Admin high: Denying request {request}")
                    request.deny("You gave me an `invalid_func` function")
                else:
                    ctx.logger.info(
                        f"Admin high: Approving request by executing {request}"
                    )
                    job = request.code(blocking=False)
                    result = job.wait()
                    ctx.logger.info(f"Admin high: Request result {result}")

        if ctx.events.is_set(Event.ADMIN_LOW_ALL_RESULTS_AVAILABLE):
            break

    ctx.logger.info("Admin high: Done approving / denying all requests")


@sim_activity(trigger=Event.ADMIN_HIGHSIDE_FLOW_COMPLETED)
async def admin_high_side(ctx: SimulatorContext, admin_auth):
    admin_client = sy.login(**admin_auth)
    ctx.logger.info("Admin high-side: logged in")

    await asyncio.gather(
        admin_create_bq_pool_high(ctx, admin_client),
        admin_create_endpoints(ctx, admin_client),
        admin_triage_requests_high(ctx, admin_client),
    )


@sim_activity(trigger=Event.ADMIN_LOWSIDE_FLOW_COMPLETED)
async def admin_low_side(ctx: SimulatorContext, admin_auth, users):
    admin_client = sy.login(**admin_auth)
    ctx.logger.info("Admin low-side: logged in")

    await asyncio.gather(
        admin_register_users(ctx, admin_client, users),
        admin_create_bq_pool_low(ctx, admin_client),
        admin_watch_sync(ctx, admin_client),
    )


# ------------------------------------------------------------------------------------------------


@sim_activity(trigger=Event.ADMIN_SYNC_COMPLETED)
async def admin_sync_to_low_flow(
    ctx: SimulatorContext, admin_auth_high: dict, admin_auth_low: dict
):
    high_client = sy.login(**admin_auth_high)
    ctx.logger.info("Admin: logged in to high-side")

    low_client = sy.login(**admin_auth_low)
    ctx.logger.info("Admin: logged in to low-side")

    while True:
        await asyncio.sleep(random.uniform(5, 10))

        result = sy.sync(high_client, low_client)
        if isinstance(result, sy.SyftSuccess):
            ctx.logger.info("Admin high: Nothing to sync high->low")
            continue

        ctx.logger.info(f"Admin high: Syncing high->low {result}")
        result._share_all()
        result._sync_all()

        # trigger an event so that guest users can start querying
        ctx.events.trigger(Event.ADMIN_SYNCED_HIGH_TO_LOW)
        ctx.logger.info("Admin high: Synced high->low")

        if ctx.events.is_set(Event.ADMIN_HIGHSIDE_FLOW_COMPLETED):
            ctx.logger.info("Admin high: Done syncing high->low")
            break


@sim_activity(trigger=Event.ADMIN_SYNC_COMPLETED)
async def admin_sync_to_high_flow(
    ctx: SimulatorContext, admin_auth_high: dict, admin_auth_low: dict
):
    high_client = sy.login(**admin_auth_high)
    ctx.logger.info("Admin low: logged in to high-side")

    low_client = sy.login(**admin_auth_low)
    ctx.logger.info("Admin low: logged in to low-side")

    while not ctx.events.is_set(Event.ADMIN_HIGHSIDE_FLOW_COMPLETED):
        await asyncio.sleep(random.uniform(5, 10))

        ctx.logger.info("Admin low: Started sy.sync low->high")
        result = sy.sync(low_client, high_client)
        if isinstance(result, sy.SyftSuccess):
            ctx.logger.info("Admin low: Nothing to sync low->high")
            continue

        ctx.logger.info(f"Admin low: sy.sync low->high result={result}")
        result._share_all()
        result._sync_all()

        ctx.events.trigger(Event.ADMIN_SYNCED_LOW_TO_HIGH)
        ctx.logger.info("Admin low: Synced low->high")

        if ctx.events.is_set(Event.ADMIN_HIGHSIDE_FLOW_COMPLETED):
            ctx.logger.info("Admin high: Done syncing high->low")
            break


# ------------------------------------------------------------------------------------------------


@sim_entrypoint
async def sim_l0_scenario(ctx: SimulatorContext):
    users = [
        dict(  # noqa: C408
            name=fake.name(),
            email=fake.email(),
            password="password",
        )
        for _ in range(NUM_USERS)
    ]

    server_url_high = "http://localhost:8080"
    admin_auth_high = dict(  # noqa: C408
        url=server_url_high,
        email="info@openmined.org",
        password="changethis",
    )

    server_url_low = "http://localhost:8081"
    admin_auth_low = dict(  # noqa: C408
        url=server_url_low,
        email="info@openmined.org",
        password="changethis",
    )

    ctx.events.trigger(Event.INIT)
    ctx.logger.info("--- Initializing L0 BigQuery Scenario Test ---")

    await asyncio.gather(
        admin_low_side(ctx, admin_auth_low, users),
        admin_high_side(ctx, admin_auth_high),
        admin_sync_to_low_flow(ctx, admin_auth_high, admin_auth_low),
        admin_sync_to_high_flow(ctx, admin_auth_high, admin_auth_low),
        *[user_flow(ctx, server_url_low, user) for user in users],
    )


@pytest.mark.asyncio
async def test_l0_scenario(request):
    sim = Simulator("l0_scenario")

    await sim.start(
        sim_l0_scenario,
        random_wait=None,  # (0.5, 1.5),
        check_events=[
            # admin lowside
            Event.GUEST_USERS_CREATED,
            Event.ADMIN_LOWSIDE_WORKER_POOL_CREATED,
            Event.ADMIN_LOW_ALL_RESULTS_AVAILABLE,
            Event.ADMIN_LOWSIDE_FLOW_COMPLETED,
            # admin high side
            Event.ADMIN_ALL_ENDPOINTS_CREATED,
            Event.ADMIN_HIGHSIDE_WORKER_POOL_CREATED,
            Event.ADMIN_HIGHSIDE_FLOW_COMPLETED,
            # admin sync
            Event.ADMIN_SYNC_COMPLETED,
            # users
            Event.USER_CAN_QUERY_TEST_ENDPOINT,
            Event.USER_CHECKED_RESULTS,
            Event.USER_FLOW_COMPLETED,
        ],
        timeout=300,
    )
