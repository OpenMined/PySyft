# third party
from api import get_datasets
from asserts import has
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


@pytest.mark.asyncio
async def test_create_dataset_and_read_mock(request):
    events = EventManager()
    server = make_server(request)

    dataset_get_all = with_client(get_datasets, server)

    assert dataset_get_all() == 0

    fake = Faker()
    admin = make_admin()
    events.register(EVENT_USER_ADMIN_CREATED)

    root_client = admin.client(server)
    dataset_name = fake.name()
    dataset = create_dataset(name=dataset_name)

    upload_dataset(root_client, dataset)

    events.register(EVENT_DATASET_UPLOADED)

    users = [make_user() for i in range(2)]

    user = users[0]

    user_can_read_mock_dataset(server, events, user, dataset_name)
    create_users(root_client, events, users)

    await has(
        lambda: dataset_get_all() == 1,
        "1 Dataset",
        timeout=15,
        retry=1,
    )

    await events.wait_for(event_name=EVENT_DATASET_MOCK_READABLE)
