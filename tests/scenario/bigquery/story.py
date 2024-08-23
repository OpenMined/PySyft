# third party
from events import EVENT_DATASET_MOCK_READABLE
from events import EVENT_USERS_CREATED
from fixtures_sync import trade_flow_df
from fixtures_sync import trade_flow_df_mock
from unsync import unsync


@unsync
async def user_can_read_mock_dataset(server, events, user, dataset_name):
    print("waiting ", EVENT_USERS_CREATED)
    await events.wait_for(event_name=EVENT_USERS_CREATED)
    user_client = user.client(server)
    print("getting dataset", dataset_name)
    mock = user_client.api.services.dataset[dataset_name].assets[0].mock
    df = trade_flow_df_mock(trade_flow_df())
    print("Are we here?")
    if df.equals(mock):
        print("REGISTERING EVENT", EVENT_DATASET_MOCK_READABLE)
        events.register(EVENT_DATASET_MOCK_READABLE)
