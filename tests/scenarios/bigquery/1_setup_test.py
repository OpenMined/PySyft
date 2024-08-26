# third party
from api import get_datasets
from asserts import has
import asyncio
from events import EVENT_DATASET_MOCK_READABLE
from events import EVENT_DATASET_UPLOADED
from events import EVENT_USER_ADMIN_CREATED
from events import EventManager
from faker import Faker
from fixtures_sync import create_dataset
from fixtures_sync import make_admin
from fixtures_sync import make_server
from fixtures_sync import make_user
from fixtures_sync import upload_dataset
from make import create_users
from partials import with_client
import pytest
from story import user_can_read_mock_dataset
from syft import test_settings
import syft as sy
from asserts import FailedAssert
from unsync import unsync

from events import EVENT_USERS_CREATED

EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED = "prebuilt_worker_image_bigquery_created"
EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED = "external_registry_bigquery_created"
EVENT_WORKER_POOL_CREATED = "worker_pool_created"
EVENT_ALLOW_GUEST_SIGNUP_DISABLED = "allow_guest_signup_disabled"
EVENT_USERS_CREATED_CHECKED = "users_created_checked"


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
def get_prebuilt_worker_image(events, client, expected_tag, event_name):
    if events.wait_for(event_name=event_name):
        worker_images = client.images.get_all()
        for worker_image in worker_images:
            if expected_tag in str(worker_image.image_identifier):
                assert expected_tag in str(worker_image.image_identifier)
                return worker_image
    raise FailedAssert(f"get_prebuilt_worker_image cannot find {expected_tag}")


async def create_prebuilt_worker_image(events, client, expected_tag, event_name):
    print("1")
    external_registry = test_settings.get("external_registry", default="docker.io")
    print("2")
    docker_config = sy.PrebuiltWorkerConfig(tag=f"{external_registry}/{expected_tag}")
    print("3")
    result = client.api.services.worker_image.submit(worker_config=docker_config)
    print("4", result)
    assert isinstance(result, sy.SyftSuccess)
    events.register(event_name)
    print("5", event_name)


@unsync
def add_external_registry(events, client, event_name):
    external_registry = test_settings.get("external_registry", default="docker.io")
    result = client.api.services.image_registry.add(external_registry)
    assert isinstance(result, sy.SyftSuccess)
    events.register(event_name)


@unsync
def create_worker_pool(
    events, client, worker_pool_name, worker_pool_result, event_name
):
    # block until this is available
    worker_image = worker_pool_result.result(timeout=5)

    result = client.api.services.worker_pool.launch(
        pool_name=worker_pool_name,
        image_uid=worker_image.id,
        num_workers=1,
    )
    assert isinstance(result, sy.SyftSuccess)
    events.register(event_name)


async def check_worker_pool_exists(events, client, worker_pool_name, event_name):
    timeout = 30
    print("waiting for check_worker_pool_exists", event_name, timeout)
    await events.wait_for(event_name=event_name, timeout=timeout)
    print("its been 30 seconds, trying to get all worker pools")
    pools = client.worker_pools.get_all()
    print("pools", len(pools), pools)
    for pool in pools:
        print("pool name", pool.name)
        if worker_pool_name == pool.name:
            assert worker_pool_name == pool.name
            return worker_pool_name == pool.name

    raise FailedAssert(
        f"check_worker_pool_exists cannot find worker_pool_name {worker_pool_name}"
    )


def set_settings_allow_guest_signup(events, client, enabled, event_name):
    result = client.settings.allow_guest_signup(enable=enabled)
    assert isinstance(result, sy.SyftSuccess)
    events.register(event_name)


async def check_users_created(events, client, users, event_name, event_set):
    print("check users created")
    expected_emails = {user.email for user in users}
    found_emails = set()
    print("wait for created event", event_name)
    await events.wait_for(event_name=event_name)
    print("finished waiting getting all the users")
    user_results = client.api.services.user.get_all()
    for user_result in user_results:
        if user_result.email in expected_emails:
            found_emails.add(user_result.email)

    print(
        "len(found_emails) == len(expected_emails)",
        len(found_emails) == len(expected_emails),
    )
    if len(found_emails) == len(expected_emails):
        events.register(event_set)
    else:
        raise FailedAssert(
            f"check_users_created only found {len(found_emails)} of {len(expected_emails)} "
            f"emails: {found_emails}, {expected_emails}"
        )


@pytest.mark.asyncio
async def test_create_dataset_and_read_mock(request):
    events = EventManager()
    server = make_server(request)

    admin = make_admin()
    events.register(EVENT_USER_ADMIN_CREATED)
    await events.wait_for(event_name=EVENT_USER_ADMIN_CREATED)
    assert events.happened(EVENT_USER_ADMIN_CREATED)

    root_client = admin.client(server)
    worker_pool_name = "bigquery-pool"

    worker_docker_tag = f"openmined/bigquery:{sy.__version__}"
    print("running create_prebuilt_worker_image")
    asyncio.create_task(
        create_prebuilt_worker_image(
            events,
            root_client,
            worker_docker_tag,
            EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
        )
    )
    print("finished queing create_prebuilt_worker_image")

    print("waiting...")
    # await events.wait_for(
    #     event_name=EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED, timeout=30
    # )
    # assert await events.happened(EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED)

    # worker_image_result = get_prebuilt_worker_image(
    #     events,
    #     root_client,
    #     worker_docker_tag,
    #     EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED,
    # )

    # await events.wait_for(event_name=EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED)
    # assert events.happened(EVENT_PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED)

    # add_external_registry(events, root_client, EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED)

    # await events.wait_for(event_name=EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED)
    # assert events.happened(EVENT_EXTERNAL_REGISTRY_BIGQUERY_CREATED)

    # create_worker_pool(
    #     events,
    #     root_client,
    #     worker_pool_name,
    #     worker_image_result,
    #     EVENT_WORKER_POOL_CREATED,
    # )

    # await events.wait_for(event_name=EVENT_WORKER_POOL_CREATED)
    # assert events.happened(EVENT_WORKER_POOL_CREATED)

    # check_worker_pool_exists(
    #     events, root_client, worker_pool_name, EVENT_WORKER_POOL_CREATED
    # )

    # await events.wait_for(event_name=EVENT_WORKER_POOL_CREATED)
    # assert events.happened(EVENT_WORKER_POOL_CREATED)

    # set_settings_allow_guest_signup(
    #     events, root_client, False, EVENT_ALLOW_GUEST_SIGNUP_DISABLED
    # )

    # await events.wait_for(event_name=EVENT_ALLOW_GUEST_SIGNUP_DISABLED)
    # assert events.happened(EVENT_ALLOW_GUEST_SIGNUP_DISABLED)

    # users = [make_user() for i in range(2)]
    # # user = users[0]

    # create_users(root_client, events, users, EVENT_USERS_CREATED)

    # await events.wait_for(event_name=EVENT_USERS_CREATED)
    # assert events.happened(EVENT_USERS_CREATED)

    # check_users_created(
    #     events, root_client, users, EVENT_USERS_CREATED, EVENT_USERS_CREATED_CHECKED
    # )

    # await events.wait_for(event_name=EVENT_USERS_CREATED_CHECKED)
    # assert events.happened(EVENT_USERS_CREATED_CHECKED)

    # check users are created
    # high_client.api.services.user.get_all()

    # # check_cant_sign_up

    # # create users

    # # create api endpoints
    # # check they respond

    # # login as user
    # # test queries
    # # submit code via api
    # # verify its not accessible yet

    # # continuously checking for
    # # new untriaged requests
    # # executing them locally
    # # submitting the results

    # # users get the results
    # # continuously checking
    # # assert he random number of rows is there

    # # await has(
    # #     lambda: dataset_get_all() == 1,
    # #     "1 Dataset",
    # #     timeout=15,
    # #     retry=1,
    # # )

    # await events.wait_for(event_name=EVENT_USERS_CREATED_CHECKED, timeout=60)
