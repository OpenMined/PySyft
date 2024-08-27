# stdlib
import asyncio
import inspect

# third party
from asserts import ensure_package_installed
from events import EVENT_ALLOW_GUEST_SIGNUP_DISABLED
from events import EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED
from events import EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED
from events import EVENT_SCHEMA_ENDPOINT_CREATED
from events import EVENT_SUBMIT_QUERY_ENDPOINT_CONFIGURED
from events import EVENT_SUBMIT_QUERY_ENDPOINT_CREATED
from events import EVENT_USERS_CAN_QUERY_MOCK
from events import EVENT_USERS_CAN_SUBMIT_QUERY
from events import EVENT_USERS_CREATED
from events import EVENT_USERS_CREATED_CHECKED
from events import EVENT_USER_ADMIN_CREATED
from events import EVENT_WORKER_POOL_CREATED
from events import EventManager
from events import Scenario
from faker import Faker
from fixtures_sync import make_admin
from fixtures_sync import make_server
from fixtures_sync import make_user
from make import create_endpoints_query
from make import create_endpoints_schema
from make import create_endpoints_submit_query
from make import create_users
import pytest
from unsync import unsync

# syft absolute
import syft as sy
from syft import test_settings

# dataset stuff
# """
# dataset_get_all = with_client(get_datasets, server)

# assert dataset_get_all() == 0

# dataset_name = fake.name()
# dataset = create_dataset(name=dataset_name)

# upload_dataset(root_client, dataset)

# events.register(EVENT_DATASET_UPLOADED)

# user_can_read_mock_dataset(server, events, user, dataset_name)

# await has(
#     lambda: dataset_get_all() == 1,
#     "1 Dataset",
#     timeout=15,
#     retry=1,
# )

# """


@unsync
async def get_prebuilt_worker_image(events, client, expected_tag, event_name):
    await events.await_for(event_name=event_name, show=True)
    worker_images = client.images.get_all()
    for worker_image in worker_images:
        if expected_tag in str(worker_image.image_identifier):
            assert expected_tag in str(worker_image.image_identifier)
            return worker_image
    print(f"get_prebuilt_worker_image cannot find {expected_tag}")
    # raise FailedAssert()


@unsync
async def create_prebuilt_worker_image(events, client, expected_tag, event_name):
    external_registry = test_settings.get("external_registry", default="docker.io")
    docker_config = sy.PrebuiltWorkerConfig(tag=f"{external_registry}/{expected_tag}")
    result = client.api.services.worker_image.submit(worker_config=docker_config)
    assert isinstance(result, sy.SyftSuccess)
    asyncio.sleep(5)
    events.register(event_name)


@unsync
async def add_external_registry(events, client, event_name):
    external_registry = test_settings.get("external_registry", default="docker.io")
    result = client.api.services.image_registry.add(external_registry)
    assert isinstance(result, sy.SyftSuccess)
    events.register(event_name)


@unsync
async def create_worker_pool(
    events, client, worker_pool_name, worker_pool_result, event_name
):
    # block until this is available
    worker_image = worker_pool_result.result(timeout=5)

    result = client.api.services.worker_pool.launch(
        pool_name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=1,
    )

    if isinstance(result, list) and isinstance(
        result[0], sy.service.worker.worker_pool.ContainerSpawnStatus
    ):
        events.register(event_name)
    else:
        print("bad result", result)


@unsync
async def check_worker_pool_exists(events, client, worker_pool_name, event_name):
    timeout = 30
    await events.await_for(event_name=event_name, timeout=timeout)
    pools = client.worker_pools.get_all()
    for pool in pools:
        if worker_pool_name == pool.name:
            assert worker_pool_name == pool.name
            return worker_pool_name == pool.name

    print(f"check_worker_pool_exists cannot find worker_pool_name {worker_pool_name}")
    # raise FailedAssert(

    # )


@unsync
async def set_settings_allow_guest_signup(
    events, client, enabled, event_name: str | None = None
):
    result = client.settings.allow_guest_signup(enable=enabled)
    if event_name:
        if isinstance(result, sy.SyftSuccess):
            events.register(event_name)
        else:
            print("cant set settings alow guest signup")


@unsync
async def check_users_created(events, client, users, event_name, event_set):
    expected_emails = {user.email for user in users}
    found_emails = set()
    await events.await_for(event_name=event_name)
    user_results = client.api.services.user.get_all()
    for user_result in user_results:
        if user_result.email in expected_emails:
            found_emails.add(user_result.email)

    if len(found_emails) == len(expected_emails):
        events.register(event_set)
    else:
        print(
            f"check_users_created only found {len(found_emails)} of {len(expected_emails)} "
            f"emails: {found_emails}, {expected_emails}"
        )
        # raise FailedAssert()


def guest_register(client, test_user):
    guest_client = client.guest()
    fake = Faker()
    result = guest_client.register(
        name=test_user.name,
        email=test_user.email,
        password=test_user.password,
        password_verify=test_user.password,
        institution=fake.company(),
        website=fake.url(),
    )
    return result


async def result_is(
    events,
    expr,
    matches: bool | str | type | object,
    after: str | None = None,
    register: str | None = None,
):
    if after:
        await events.await_for(event_name=after)

    lambda_source = inspect.getsource(expr)
    try:
        result = expr()
        assertion = False
        if isinstance(matches, bool):
            assertion = result == matches
        elif isinstance(matches, type):
            assertion = isinstance(result, matches)
        elif isinstance(matches, str):
            message = matches.replace("*", "")
            assertion = message in str(result)
        else:
            if hasattr(result, "message"):
                message = result.message.replace("*", "")
                assertion = isinstance(result, type(matches)) and message in str(result)

        if assertion and register:
            events.register(event_name=register)
        return assertion
    except Exception as e:
        print(f"insinstance({lambda_source}, {matches}). {e}")

    return False


@unsync
async def set_endpoint_settings(
    events, client, path, kwargs, after: str, register: str
):
    if after:
        await events.await_for(event_name=after)

    # Here, we update the endpoint to timeout after 100s (rather the default of 60s)
    result = client.api.services.api.update(endpoint_path=path, **kwargs)
    if isinstance(result, sy.SyftSuccess):
        events.register(register)
    else:
        print(f"Failed to update api endpoint. {path}")


EVENT_QUERY_ENDPOINT_CREATED = "query_endpoint_created"
EVENT_QUERY_ENDPOINT_CONFIGURED = "query_endpoint_configured"


def query_sql():
    query = f"SELECT {test_settings.table_2_col_id}, AVG({test_settings.table_2_col_score}) AS average_score \
        FROM {test_settings.dataset_2}.{test_settings.table_2} \
        GROUP BY {test_settings.table_2_col_id} \
        LIMIT 10000"
    return query


def run_code(client, method_name, **kwargs):
    service_func_name = method_name
    if "*" in method_name:
        matcher = method_name.replace("*", "")
        all_code = client.api.services.code.get_all()
        for code in all_code:
            if matcher in code.service_func_name:
                service_func_name = code.service_func_name
                break

    api_method = api_for_path(client, path=f"code.{service_func_name}")
    result = api_method(**kwargs)
    return result


def run_api_path(client, path, **kwargs):
    api_method = api_for_path(client, path)
    result = api_method(**kwargs)
    return result


def api_for_path(client, path):
    root = client.api.services
    for part in path.split("."):
        if hasattr(root, part):
            root = getattr(root, part)
        else:
            print("cant find part", part, path)
    return root


EVENT_USERS_QUERY_NOT_READY = "users_query_not_ready"


@pytest.mark.asyncio
async def test_level_2_basic_scenario(request):
    ensure_package_installed("google-cloud-bigquery", "google.cloud.bigquery")
    ensure_package_installed("db-dtypes", "db_dtypes")

    scenario = Scenario(
        name="test_create_dataset_and_read_mock",
        events=[
            EVENT_USER_ADMIN_CREATED,
            EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
            EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED,
            EVENT_WORKER_POOL_CREATED,
            EVENT_ALLOW_GUEST_SIGNUP_DISABLED,
            EVENT_USERS_CREATED,
            EVENT_USERS_CREATED_CHECKED,
            EVENT_QUERY_ENDPOINT_CREATED,
            EVENT_QUERY_ENDPOINT_CONFIGURED,
            EVENT_SCHEMA_ENDPOINT_CREATED,
            EVENT_SUBMIT_QUERY_ENDPOINT_CREATED,
            EVENT_SUBMIT_QUERY_ENDPOINT_CONFIGURED,
            EVENT_USERS_CAN_QUERY_MOCK,
            EVENT_USERS_CAN_SUBMIT_QUERY,
            EVENT_USERS_QUERY_NOT_READY,
        ],
    )

    events = EventManager()
    events.add_scenario(scenario)
    events.monitor()

    server = make_server(request)

    admin = make_admin()
    events.register(EVENT_USER_ADMIN_CREATED)

    await events.await_for(event_name=EVENT_USER_ADMIN_CREATED)
    assert events.happened(EVENT_USER_ADMIN_CREATED)

    root_client = admin.client(server)
    worker_pool_name = "bigquery-pool"

    worker_docker_tag = f"openmined/bigquery:{sy.__version__}"

    create_prebuilt_worker_image(
        events,
        root_client,
        worker_docker_tag,
        EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
    )

    worker_image_result = get_prebuilt_worker_image(
        events,
        root_client,
        worker_docker_tag,
        EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
    )

    add_external_registry(events, root_client, EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED)

    create_worker_pool(
        events,
        root_client,
        worker_pool_name,
        worker_image_result,
        EVENT_WORKER_POOL_CREATED,
    )

    check_worker_pool_exists(
        events, root_client, worker_pool_name, EVENT_WORKER_POOL_CREATED
    )

    set_settings_allow_guest_signup(
        events, root_client, False, EVENT_ALLOW_GUEST_SIGNUP_DISABLED
    )

    users = [make_user() for i in range(2)]

    create_users(root_client, events, users, EVENT_USERS_CREATED)

    check_users_created(
        events, root_client, users, EVENT_USERS_CREATED, EVENT_USERS_CREATED_CHECKED
    )

    create_endpoints_query(
        events,
        root_client,
        worker_pool_name=worker_pool_name,
        register=EVENT_QUERY_ENDPOINT_CREATED,
    )

    test_query_path = "bigquery.test_query"
    set_endpoint_settings(
        events,
        root_client,
        path=test_query_path,
        kwargs={"endpoint_timeout": 120, "hide_mock_definition": True},
        after=EVENT_QUERY_ENDPOINT_CREATED,
        register=EVENT_QUERY_ENDPOINT_CONFIGURED,
    )

    print("calling create endpoints schema")
    create_endpoints_schema(
        events,
        root_client,
        worker_pool_name=worker_pool_name,
        register=EVENT_SCHEMA_ENDPOINT_CREATED,
    )

    create_endpoints_submit_query(
        events,
        root_client,
        worker_pool_name=worker_pool_name,
        register=EVENT_SUBMIT_QUERY_ENDPOINT_CREATED,
    )

    submit_query_path = "bigquery.submit_query"
    set_endpoint_settings(
        events,
        root_client,
        path=submit_query_path,
        kwargs={"hide_mock_definition": True},
        after=EVENT_SUBMIT_QUERY_ENDPOINT_CREATED,
        register=EVENT_SUBMIT_QUERY_ENDPOINT_CONFIGURED,
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
        after=[EVENT_QUERY_ENDPOINT_CONFIGURED, EVENT_USERS_CREATED_CHECKED],
        register=EVENT_USERS_CAN_QUERY_MOCK,
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
        after=[EVENT_SUBMIT_QUERY_ENDPOINT_CONFIGURED, EVENT_USERS_CREATED_CHECKED],
        register=EVENT_USERS_CAN_SUBMIT_QUERY,
    )

    await result_is(
        events,
        lambda: run_code(users[0].client(server), method_name=f"{func_name}*"),
        matches=sy.SyftError(message="*Your code is waiting for approval*"),
        after=[EVENT_USERS_CAN_SUBMIT_QUERY],
        register=EVENT_USERS_QUERY_NOT_READY,
    )

    # # continuously checking for
    # # new untriaged requests
    # # executing them locally
    # # submitting the results

    # # users get the results
    # # continuously checking
    # # assert he random number of rows is there

    res = await result_is(
        events,
        lambda: guest_register(root_client, make_user()),
        matches=sy.SyftError(
            message="*You don't have permission to create an account*"
        ),
        after=EVENT_ALLOW_GUEST_SIGNUP_DISABLED,
    )

    assert res is True

    await events.await_scenario(scenario_name="test_create_dataset_and_read_mock")
    assert events.scenario_completed("test_create_dataset_and_read_mock")
