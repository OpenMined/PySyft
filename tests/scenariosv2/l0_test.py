# stdlib
import asyncio
from enum import auto
import os
import random

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.orchestra import DeploymentType
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
from .flows.utils import launch_server
from .sim.core import BaseEvent
from .sim.core import Simulator
from .sim.core import SimulatorContext
from .sim.core import sim_activity
from .sim.core import sim_entrypoint

fake = Faker()
NUM_USERS = 10
NUM_ENDPOINTS = 3  # test_query, submit_query, schema_query
TIMEOUT = 900


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
async def user_low_side_flow(ctx: SimulatorContext, server_url_low: str, user: dict):
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
async def admin_high_create_endpoints(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    worker_pool = "biquery-pool"

    await asyncio.gather(
        admin_create_bq_test_endpoint(ctx, admin_client, worker_pool),
        admin_create_bq_submit_endpoint(ctx, admin_client, worker_pool),
        admin_create_bq_schema_endpoint(ctx, admin_client, worker_pool),
    )
    ctx.logger.info("Admin high: Created all endpoints")


def all_available(paths: list[str], expected: list[str]):
    return set(expected).issubset(set(paths))


@sim_activity(
    # endpoints work only after low side worker pool is created
    wait_for=Event.ADMIN_LOWSIDE_WORKER_POOL_CREATED
)
async def admin_low_triage_requests(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    expected_paths = [
        "bigquery.test_query",
        "bigquery.submit_query",
        "bigquery.schema",
    ]

    while True:
        await asyncio.sleep(random.uniform(5, 10))

        # check if endpoints are available
        if not ctx.events.is_set(Event.ADMIN_LOW_SIDE_ENDPOINTS_AVAILABLE):
            endpoints = admin_client.custom_api.get_all()
            paths = [ep.path for ep in endpoints]
            ctx.logger.debug(f"Admin low: API endpoints - {paths}")

            if all_available(paths, expected_paths):
                ctx.logger.info("Admin low: All endpoints available")
                ctx.events.trigger(Event.ADMIN_LOW_SIDE_ENDPOINTS_AVAILABLE)
            else:
                ctx.logger.info(f"Admin low: Waiting for all endpoints {paths}")

        # Check if all requests are approved or denied
        requests = admin_client.requests.get_all()
        pending = [req for req in requests if req.status == RequestStatus.PENDING]
        ctx.logger.info(f"Admin low: Requests={len(requests)} Pending={len(pending)}")

        # If all requests have been triaged, then exit
        if len(requests) == NUM_USERS:
            ctx.events.trigger(Event.ADMIN_LOW_ALL_RESULTS_AVAILABLE)
            break

    ctx.logger.info("Admin low: All requests triaged.")


@sim_activity(trigger=Event.ADMIN_HIGHSIDE_WORKER_POOL_CREATED)
async def admin_high_create_bq_pool(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    await asyncio.to_thread(bq_create_pool, ctx, admin_client)


@sim_activity(trigger=Event.ADMIN_LOWSIDE_WORKER_POOL_CREATED)
async def admin_low_create_bq_pool(
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
async def admin_high_triage_requests(
    ctx: SimulatorContext, admin_client: sy.DatasiteClient
):
    while not ctx.events.is_set(Event.ADMIN_LOW_ALL_RESULTS_AVAILABLE):
        await asyncio.sleep(random.uniform(5, 10))

        # check if there are any requests
        # BUG: request that are executed request.code() are always in pending state
        requests = admin_client.requests.get_all()
        pending = [req for req in requests if req.status == RequestStatus.PENDING]
        ctx.logger.info(f"Admin high: Requests={len(requests)} Pending={len(pending)}")

        for request in pending:
            # ignore non-code requests
            if not getattr(request, "code", None):
                continue

            if "invalid_func" in request.code.service_func_name:
                ctx.logger.info(f"Admin high: Denying request {request}")
                request.deny("You gave me an `invalid_func` function")
            else:
                ctx.logger.info(f"Admin high: Approving request by executing {request}")
                func_name = request.code.service_func_name
                api_func = getattr(admin_client.code, func_name, None)
                job = api_func(blocking=False)
                result = job.wait()
                ctx.logger.info(f"Admin high: Request result {result}")
        if len(requests) == NUM_USERS:
            break
    ctx.logger.info("Admin high: All requests triaged.")


@sim_activity(trigger=Event.ADMIN_HIGHSIDE_FLOW_COMPLETED)
async def admin_high_side_flow(ctx: SimulatorContext, admin_auth):
    admin_client = sy.login(**admin_auth)
    ctx.logger.info("Admin high: logged in")

    await asyncio.gather(
        admin_high_create_bq_pool(ctx, admin_client),
        admin_high_create_endpoints(ctx, admin_client),
        admin_high_triage_requests(ctx, admin_client),
    )


@sim_activity(trigger=Event.ADMIN_LOWSIDE_FLOW_COMPLETED)
async def admin_low_side_flow(ctx: SimulatorContext, admin_auth, users):
    admin_client = sy.login(**admin_auth)
    ctx.logger.info("Admin low: logged in")

    await asyncio.gather(
        admin_register_users(ctx, admin_client, users),
        admin_low_create_bq_pool(ctx, admin_client),
        admin_low_triage_requests(ctx, admin_client),
    )


# ------------------------------------------------------------------------------------------------


async def admin_sync(
    ctx: SimulatorContext,
    from_auth: dict,
    to_auth: dict,
    trigger: Event,
    exit_after: Event,
):
    from_client = sy.login(**from_auth)
    to_client = sy.login(**to_auth)

    from_ = from_client.metadata.server_side_type
    to_ = to_client.metadata.server_side_type

    while not ctx.events.is_set(exit_after):
        try:
            await asyncio.sleep(random.uniform(3, 5))

            ctx.logger.info(f"Admin {from_}: Sync {from_}->{to_} - Checking")
            result = sy.sync(from_client, to_client)
            if isinstance(result, sy.SyftSuccess):
                continue

            ctx.logger.info(f"Admin {from_}: Sync {from_}->{to_} - Result={result}")
            result._share_all()
            result._sync_all()

            ctx.events.trigger(trigger)
            ctx.logger.info(f"Admin {from_}: Sync {from_}->{to_} - Synced")

        except Exception as e:
            ctx.logger.error(f"Admin {from_}: Sync {from_}->{to_} - Error: {str(e)}")
            ctx.logger.info(f"Admin {from_}: Sync {from_}->{to_} - Waiting a bit..")
            await asyncio.sleep(random.uniform(2, 4))

    ctx.logger.info(f"Admin {from_}: Sync {from_}->{to_} - Closed")


@sim_activity(trigger=Event.ADMIN_SYNC_COMPLETED)
async def admin_sync_high_to_low_flow(
    ctx: SimulatorContext, admin_auth_high: dict, admin_auth_low: dict
):
    await admin_sync(
        ctx,
        # high -> low
        from_auth=admin_auth_high,
        to_auth=admin_auth_low,
        trigger=Event.ADMIN_SYNCED_HIGH_TO_LOW,
        # TODO: see if we have a better exit clause
        exit_after=Event.ADMIN_HIGHSIDE_FLOW_COMPLETED,
    )


@sim_activity(trigger=Event.ADMIN_SYNC_COMPLETED)
async def admin_sync_low_to_high_flow(
    ctx: SimulatorContext, admin_auth_high: dict, admin_auth_low: dict
):
    await admin_sync(
        ctx,
        # low -> high
        from_auth=admin_auth_low,
        to_auth=admin_auth_high,
        trigger=Event.ADMIN_SYNCED_LOW_TO_HIGH,
        # TODO: see if we have a better exit clause
        exit_after=Event.ADMIN_LOWSIDE_FLOW_COMPLETED,
    )


# ------------------------------------------------------------------------------------------------


def setup_servers(ctx: SimulatorContext, server_url_high, server_url_low):
    deployment_type = os.environ.get("ORCHESTRA_DEPLOYMENT_TYPE", DeploymentType.REMOTE)
    ctx.logger.info(f"Deployment type: {deployment_type}")

    if deployment_type == DeploymentType.REMOTE:
        return None, None

    ctx.logger.info(f"Launching python server high side server on {server_url_high}")
    server_high = launch_server(
        server_url=server_url_high,
        server_name="syft-high",
        server_side_type="high",
    )

    ctx.logger.info(f"Launching python server low side server on {server_url_low}")
    server_low = launch_server(
        server_url=server_url_low,
        server_name="syft-low",
        server_side_type="low",
    )

    return server_high, server_low


def shutdown_servers(server_high, server_low):
    if server_high:
        server_high.land()

    if server_low:
        server_low.land()


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

    server_high, server_low = setup_servers(ctx, server_url_high, server_url_low)

    ctx.events.trigger(Event.INIT)
    ctx.logger.info("--- Initializing L0 BigQuery Scenario Test ---")

    await asyncio.gather(
        admin_low_side_flow(ctx, admin_auth_low, users),
        admin_high_side_flow(ctx, admin_auth_high),
        admin_sync_high_to_low_flow(ctx, admin_auth_high, admin_auth_low),
        admin_sync_low_to_high_flow(ctx, admin_auth_high, admin_auth_low),
        *[user_low_side_flow(ctx, server_url_low, user) for user in users],
    )

    shutdown_servers(server_high, server_low)


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
        timeout=TIMEOUT,
    )
