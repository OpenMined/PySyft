# stdlib
import asyncio
import os
import random

# third party
from faker import Faker
import pytest

# syft absolute
import syft as sy
from syft.orchestra import DeploymentType

# relative
from .flows.user_bigquery_api import bq_submit_query
from .flows.user_bigquery_api import bq_test_query
from .flows.utils import launch_server
from .l0_test import Event
from .l0_test import admin_high_create_bq_pool
from .l0_test import admin_high_create_endpoints
from .l0_test import admin_register_users
from .sim.core import Simulator
from .sim.core import SimulatorContext
from .sim.core import sim_activity
from .sim.core import sim_entrypoint

fake = Faker()


# ---------------------------------- admin ----------------------------------
@sim_activity(
    wait_for=[
        Event.USER_CAN_SUBMIT_QUERY,
    ]
)
async def admin_triage_requests(ctx: SimulatorContext, admin_client: sy.DatasiteClient):
    while True:
        await asyncio.sleep(random.uniform(3, 5))
        ctx.logger.info("Admin: Triaging requests")

        pending_requests = admin_client.requests.get_all_pending()
        if len(pending_requests) == 0:
            break
        for request in admin_client.requests.get_all_pending():
            ctx.logger.info(f"Admin: Found request {request.__dict__}")
            if "invalid_func" in request.code.service_func_name:
                request.deny(reason="you submitted an invalid code")
            else:
                request.approve()


@sim_activity(trigger=Event.ADMIN_HIGHSIDE_FLOW_COMPLETED)
async def admin_flow(
    ctx: SimulatorContext, admin_auth: dict, users: list[dict]
) -> None:
    admin_client = sy.login(**admin_auth)
    ctx.logger.info("Admin: logged in")

    await asyncio.gather(
        admin_register_users(ctx, admin_client, users),
        admin_high_create_bq_pool(ctx, admin_client),
        admin_high_create_endpoints(ctx, admin_client),
        admin_triage_requests(ctx, admin_client),
    )


# ---------------------------------- user ----------------------------------
@sim_activity(
    wait_for=[
        Event.ADMIN_ALL_ENDPOINTS_CREATED,
        Event.ADMIN_HIGHSIDE_WORKER_POOL_CREATED,
    ],
    trigger=Event.USER_CAN_QUERY_TEST_ENDPOINT,
)
async def user_bq_test_query(ctx: SimulatorContext, client: sy.DatasiteClient):
    """Run query on test endpoint"""
    await asyncio.to_thread(bq_test_query, ctx, client)


@sim_activity(
    wait_for=[
        Event.ADMIN_ALL_ENDPOINTS_CREATED,
        Event.ADMIN_HIGHSIDE_WORKER_POOL_CREATED,
    ],
    trigger=Event.USER_CAN_SUBMIT_QUERY,
)
async def user_bq_submit_query(ctx: SimulatorContext, client: sy.DatasiteClient):
    """Submit query to be run on private data"""
    await asyncio.to_thread(bq_submit_query, ctx, client)


@sim_activity(
    wait_for=[Event.GUEST_USERS_CREATED, Event.ADMIN_ALL_ENDPOINTS_CREATED],
    trigger=Event.USER_FLOW_COMPLETED,
)
async def user_flow(ctx: SimulatorContext, server_url: str, user: dict):
    client = sy.login(
        url=server_url,
        email=user["email"],
        password=user["password"],
    )
    ctx.logger.info(f"User: {client.logged_in_user} - logged in")

    await user_bq_test_query(ctx, client)
    await user_bq_submit_query(ctx, client)


# ---------------------------------- test ----------------------------------


@sim_entrypoint
async def sim_l2_scenario(ctx: SimulatorContext):
    ctx.events.trigger(Event.INIT)
    ctx.logger.info("--- Initializing L2 BigQuery Scenario Test ---")

    users = [
        {
            "name": fake.name(),
            "email": fake.email(),
            "password": "password",
        }
        for i in range(3)
    ]

    server_url = "http://localhost:8080"
    deployment_type = os.environ.get("ORCHESTRA_DEPLOYMENT_TYPE", DeploymentType.REMOTE)
    ctx.logger.info(f"Deployment type: {deployment_type}")
    if deployment_type == DeploymentType.PYTHON:
        server = launch_server(server_url, "syft-high")

    admin_auth = {
        "url": server_url,
        "email": "info@openmined.org",
        "password": "changethis",
    }

    await asyncio.gather(
        admin_flow(ctx, admin_auth, users),
        *[user_flow(ctx, server_url, user) for user in users],
    )

    if deployment_type == DeploymentType.PYTHON:
        server.land()


@pytest.mark.asyncio
async def test_l2_scenario(request):
    sim = Simulator("l2_scenario")

    await sim.start(
        sim_l2_scenario,
        random_wait=None,
        check_events=[
            Event.GUEST_USERS_CREATED,
            Event.ADMIN_HIGHSIDE_WORKER_POOL_CREATED,
            Event.ADMIN_ALL_ENDPOINTS_CREATED,
            Event.ADMIN_HIGHSIDE_FLOW_COMPLETED,
        ],
        timeout=300,
    )
