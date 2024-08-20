# pip install pytest-asyncio pytest-timeout
# stdlib
import asyncio

# third party
from faker import Faker
from fixtures import *
import pytest


# An async function that returns "Hello, World!"
async def hello_world():
    await asyncio.sleep(1)  # Simulate some async work
    return "Hello, World!"


# # An async test function using pytest-asyncio
# @pytest.mark.asyncio
# async def test_hello_world():
#     result = await hello_world()
#     assert result == "Hello, World!"


@pytest.mark.asyncio
async def run_mock_dataframe_scenario(manager, set_event: bool = True):
    manager.reset_test_state()

    USERS_CREATED = "users_created"
    MOCK_READABLE = "mock_readable"

    fake = Faker()

    admin = make_admin()

    server = make_server(admin)

    root_client = admin.client(server)

    dataset_name = fake.name()
    dataset = create_dataset(name=dataset_name)

    result = await hello_world()
    assert result == "Hello, World!"

    upload_dataset(root_client, dataset)

    users = [make_user() for i in range(2)]

    def create_users(root_client, manager, users):
        for test_user in users:
            create_user(root_client, test_user)
        manager.register_event(USERS_CREATED)

    def user_can_read_mock_dataset(server, manager, user, dataset_name):
        print("waiting ", USERS_CREATED)
        with WaitForEvent(manager, USERS_CREATED, retry_secs=1):
            print("logging in user")
            user_client = user.client(server)
            print("getting dataset", dataset_name)
            mock = user_client.api.services.dataset[dataset_name].assets[0].mock
            df = trade_flow_df_mock(trade_flow_df())
            assert df.equals(mock)
            if set_event:
                manager.register_event(MOCK_READABLE)

    user = users[0]

    asyncit(
        user_can_read_mock_dataset,
        server=server,
        manager=manager,
        user=user,
        dataset_name=dataset_name,
    )

    asyncit(create_users, root_client=root_client, manager=manager, users=users)

    server.land()


@pytest.mark.asyncio
async def test_can_read_mock_dataframe():
    manager = TestEventManager()
    MOCK_READABLE = "mock_readable"
    await run_mock_dataframe_scenario(manager)

    async with AsyncWaitForEvent(manager, MOCK_READABLE, retry_secs=1, timeout_secs=10):
        print("Test Complete")
        result = manager.get_event_or_raise(MOCK_READABLE)
        assert result

    loop = asyncio.get_event_loop()
    loop.stop()


# @pytest.mark.asyncio
# async def test_cant_read_mock_dataframe():
#     manager = TestEventManager()
#     MOCK_READABLE = "mock_readable"
#     await run_mock_dataframe_scenario(manager, set_event=False)

#     async with AsyncWaitForEvent(manager, MOCK_READABLE, retry_secs=1, timeout_secs=10):
#         print("Test Complete")
#         with pytest.raises(Exception):
#             result = manager.get_event_or_raise(MOCK_READABLE)

#     loop = asyncio.get_event_loop()
#     loop.stop()
