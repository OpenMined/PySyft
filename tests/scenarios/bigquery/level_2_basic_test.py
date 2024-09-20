# third party
from helpers.api import create_endpoints_query
from helpers.api import create_endpoints_schema
from helpers.api import create_endpoints_submit_query
from helpers.api import query_sql
from helpers.api import run_api_path
from helpers.api import set_endpoint_settings
from helpers.asserts import ensure_package_installed
from helpers.asserts import result_is
from helpers.code import get_results
from helpers.code import run_code
from helpers.code import triage_requests
from helpers.events import Event
from helpers.events import EventManager
from helpers.events import Scenario
from helpers.fixtures_sync import create_users
from helpers.fixtures_sync import make_admin
from helpers.fixtures_sync import make_server
from helpers.fixtures_sync import make_user
from helpers.users import check_users_created
from helpers.users import guest_register
from helpers.users import set_settings_allow_guest_signup
from helpers.workers import add_external_registry
from helpers.workers import check_worker_pool_exists
from helpers.workers import create_prebuilt_worker_image
from helpers.workers import create_worker_pool
from helpers.workers import get_prebuilt_worker_image
import pytest

# syft absolute
import syft as sy


@pytest.mark.asyncio
async def test_level_2_basic_scenario(request):
    ensure_package_installed("google-cloud-bigquery", "google.cloud.bigquery")
    ensure_package_installed("db-dtypes", "db_dtypes")

    scenario = Scenario(
        name="test_create_apis_and_triage_requests",
        events=[
            Event.USER_ADMIN_CREATED,
            Event.PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
            Event.EXTERNAL_REGISTRY_BIGQUERY_CREATED,
            Event.WORKER_POOL_CREATED,
            Event.ALLOW_GUEST_SIGNUP_DISABLED,
            Event.USERS_CREATED,
            Event.USERS_CREATED_CHECKED,
            Event.QUERY_ENDPOINT_CREATED,
            Event.QUERY_ENDPOINT_CONFIGURED,
            Event.SCHEMA_ENDPOINT_CREATED,
            Event.SUBMIT_QUERY_ENDPOINT_CREATED,
            Event.SUBMIT_QUERY_ENDPOINT_CONFIGURED,
            Event.USERS_CAN_QUERY_MOCK,
            Event.USERS_CAN_SUBMIT_QUERY,
            Event.USERS_QUERY_NOT_READY,
            Event.ADMIN_APPROVED_FIRST_REQUEST,
            Event.USERS_CAN_GET_APPROVED_RESULT,
        ],
    )

    events = EventManager()
    events.add_scenario(scenario)
    events.monitor()

    server = make_server(request)

    admin = make_admin()
    events.register(Event.USER_ADMIN_CREATED)

    await events.await_for(event_name=Event.USER_ADMIN_CREATED)
    assert events.happened(Event.USER_ADMIN_CREATED)

    root_client = admin.client(server)
    triage_requests(
        events,
        root_client,
        after=Event.USER_ADMIN_CREATED,
        register=Event.ADMIN_APPROVED_FIRST_REQUEST,
    )

    worker_pool_name = "bigquery-pool"

    worker_docker_tag = f"openmined/bigquery:{sy.__version__}"

    create_prebuilt_worker_image(
        events,
        root_client,
        worker_docker_tag,
        Event.PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
    )

    worker_image_result = get_prebuilt_worker_image(
        events,
        root_client,
        worker_docker_tag,
        Event.PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
    )

    add_external_registry(events, root_client, Event.EXTERNAL_REGISTRY_BIGQUERY_CREATED)

    create_worker_pool(
        events,
        root_client,
        worker_pool_name,
        worker_image_result,
        Event.WORKER_POOL_CREATED,
    )

    check_worker_pool_exists(
        events, root_client, worker_pool_name, Event.WORKER_POOL_CREATED
    )

    set_settings_allow_guest_signup(
        events, root_client, False, Event.ALLOW_GUEST_SIGNUP_DISABLED
    )

    users = [make_user() for i in range(2)]

    create_users(root_client, events, users, Event.USERS_CREATED)

    check_users_created(
        events, root_client, users, Event.USERS_CREATED, Event.USERS_CREATED_CHECKED
    )

    create_endpoints_query(
        events,
        root_client,
        worker_pool_name=worker_pool_name,
        register=Event.QUERY_ENDPOINT_CREATED,
    )

    test_query_path = "bigquery.test_query"
    set_endpoint_settings(
        events,
        root_client,
        path=test_query_path,
        kwargs={"endpoint_timeout": 120, "hide_mock_definition": True},
        after=Event.QUERY_ENDPOINT_CREATED,
        register=Event.QUERY_ENDPOINT_CONFIGURED,
    )

    create_endpoints_schema(
        events,
        root_client,
        worker_pool_name=worker_pool_name,
        register=Event.SCHEMA_ENDPOINT_CREATED,
    )

    create_endpoints_submit_query(
        events,
        root_client,
        worker_pool_name=worker_pool_name,
        register=Event.SUBMIT_QUERY_ENDPOINT_CREATED,
    )

    submit_query_path = "bigquery.submit_query"
    set_endpoint_settings(
        events,
        root_client,
        path=submit_query_path,
        kwargs={"hide_mock_definition": True},
        after=Event.SUBMIT_QUERY_ENDPOINT_CREATED,
        register=Event.SUBMIT_QUERY_ENDPOINT_CONFIGURED,
    )

    await result_is(
        events,
        lambda: len(
            run_api_path(
                users[0].client(server), test_query_path, sql_query=query_sql()
            )
        )
        == 10000,
        matches=True,
        after=[Event.QUERY_ENDPOINT_CONFIGURED, Event.USERS_CREATED_CHECKED],
        register=Event.USERS_CAN_QUERY_MOCK,
    )

    func_name = "test_func"

    await result_is(
        events,
        lambda: run_api_path(
            users[0].client(server),
            submit_query_path,
            func_name=func_name,
            query=query_sql(),
        ),
        matches="*Query submitted*",
        after=[Event.SUBMIT_QUERY_ENDPOINT_CONFIGURED, Event.USERS_CREATED_CHECKED],
        register=Event.USERS_CAN_SUBMIT_QUERY,
    )

    await result_is(
        events,
        lambda: run_code(users[0].client(server), method_name=f"{func_name}*"),
        matches=sy.SyftException(public_message="*Your code is waiting for approval*"),
        after=[Event.USERS_CAN_SUBMIT_QUERY],
        register=Event.USERS_QUERY_NOT_READY,
    )

    get_results(
        events,
        users[0].client(server),
        method_name=f"{func_name}*",
        after=Event.USERS_QUERY_NOT_READY,
        register=Event.USERS_CAN_GET_APPROVED_RESULT,
    )

    res = await result_is(
        events,
        lambda: guest_register(root_client, make_user()),
        matches=sy.SyftException(
            public_message="*You have no permission to create an account*"
        ),
        after=Event.ALLOW_GUEST_SIGNUP_DISABLED,
    )

    assert res is True

    await events.await_scenario(
        scenario_name="test_create_apis_and_triage_requests",
        timeout=30,
    )
    assert events.scenario_completed("test_create_apis_and_triage_requests")
