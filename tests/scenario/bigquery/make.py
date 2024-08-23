# third party
from events import EVENT_USERS_CREATED
from fixtures_sync import create_user
from unsync import unsync


@unsync
async def create_users(root_client, events, users):
    for test_user in users:
        create_user(root_client, test_user)
    events.register(EVENT_USERS_CREATED)
