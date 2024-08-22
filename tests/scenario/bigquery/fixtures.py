# stdlib
import asyncio
from collections.abc import Callable
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
import json
import os
from threading import Event as ThreadingEvent
import time
from typing import Any

# third party
from faker import Faker
import pandas as pd

# syft absolute
import syft as sy
from syft import autocache
from syft.service.user.user_roles import ServiceRole

loop = None


def get_or_create_event_loop():
    try:
        # Try to get the current running event loop
        loop = asyncio.get_running_loop()
    except RuntimeError:
        # No running event loop, so create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


loop = get_or_create_event_loop()
loop.set_debug(True)


def make_server(test_user_admin):
    server = sy.orchestra.launch(
        name="test-datasite-1", port="auto", dev_mode=True, reset=True
    )
    return server


def get_root_client(server, test_user_admin):
    return server.login(email=test_user_admin.email, password=test_user_admin.password)


def trade_flow_df():
    canada_dataset_url = "https://github.com/OpenMined/datasets/blob/main/trade_flow/ca%20-%20feb%202021.csv?raw=True"
    df = pd.read_csv(autocache(canada_dataset_url))
    return df


def trade_flow_df_mock(df):
    return df[10:20]


def create_dataset(name: str):
    df = trade_flow_df()
    ca_data = df[0:10]
    mock_ca_data = trade_flow_df_mock(df)
    dataset = sy.Dataset(name=name)
    dataset.set_description("Canada Trade Data Markdown Description")
    dataset.set_summary("Canada Trade Data Short Summary")
    dataset.add_citation("Person, place or thing")
    dataset.add_url("https://github.com/OpenMined/datasets/tree/main/trade_flow")
    dataset.add_contributor(
        name="Andrew Trask",
        email="andrew@openmined.org",
        note="Andrew runs this datasite and prepared the dataset metadata.",
    )
    dataset.add_contributor(
        name="Madhava Jay",
        email="madhava@openmined.org",
        note="Madhava tweaked the description to add the URL because Andrew forgot.",
    )
    ctf = sy.Asset(name="canada_trade_flow")
    ctf.set_description(
        "Canada trade flow represents export & import of different commodities to other countries"
    )
    ctf.add_contributor(
        name="Andrew Trask",
        email="andrew@openmined.org",
        note="Andrew runs this datasite and prepared the asset.",
    )
    ctf.set_obj(ca_data)
    ctf.set_shape(ca_data.shape)
    ctf.set_mock(mock_ca_data, mock_is_real=False)
    dataset.add_asset(ctf)
    return dataset


def dataset_exists(root_client, dataset_name: str) -> bool:
    datasets = root_client.api.services.dataset
    for dataset in datasets:
        if dataset.name == dataset_name:
            return True
    return False


def upload_dataset(user_client, dataset):
    if not dataset_exists(user_client, dataset):
        user_client.upload_dataset(dataset)
    else:
        print("Dataset already exists")


@dataclass
class TestUser:
    name: str
    email: str
    password: str
    role: ServiceRole
    server_cache: Any | None = None

    def client(self, server=None):
        if server is None:
            server = self.server_cache
        else:
            self.server_cache = server

        return server.login(email=self.email, password=self.password)


def user_exists(root_client, email: str) -> bool:
    users = root_client.api.services.user
    for user in users:
        if user.email == email:
            return True
    return False


def make_user(
    name: str | None = None,
    email: str | None = None,
    password: str | None = None,
    role: ServiceRole = ServiceRole.DATA_SCIENTIST,
):
    fake = Faker()
    if name is None:
        name = fake.name()
    if email is None:
        email = fake.email()
    if password is None:
        password = fake.password()

    return TestUser(name=name, email=email, password=password, role=role)


def make_admin(email="info@openmined.org", password="changethis"):
    fake = Faker()
    return make_user(
        email=email, password=password, name=fake.name(), role=ServiceRole.ADMIN
    )


def create_user(root_client, test_user):
    if not user_exists(root_client, test_user.email):
        fake = Faker()
        root_client.register(
            name=test_user.name,
            email=test_user.email,
            password=test_user.password,
            password_verify=test_user.password,
            institution=fake.company(),
            website=fake.url(),
        )
    else:
        print("User already exists", test_user)


@dataclass
class TestEvent:
    name: str
    event_time: float = field(default_factory=lambda: time.time())

    def __post_init__(self):
        self.event_time = float(self.event_time)

    def __repr__(self):
        formatted_time = datetime.fromtimestamp(self.event_time).strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        return f"TestEvent(name={self.name}, event_time={formatted_time})"


class TestEventManager:
    def __init__(self, test_name: str):
        self.file_path = f"events_{test_name}.json"
        self._load_events()

    def _load_events(self):
        if os.path.exists(self.file_path):
            with open(self.file_path) as f:
                self.data = json.load(f)
        else:
            self.data = {"events": {}}

    def _save_events(self):
        with open(self.file_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def reset_test_state(self):
        self.data = {"events": {}}
        self._save_events()

    def register_event(self, event: TestEvent | str):
        if isinstance(event, str):
            event = TestEvent(name=event)
        if event.name not in self.data["events"]:
            self.data["events"][event.name] = asdict(event)
            self._save_events()
        else:
            print(
                f"Event '{event.name}' already exists. Use register_event_once or reset_test_state first."
            )

    def register_event_once(self, event: TestEvent):
        if event.name in self.data["events"]:
            raise ValueError(f"Event '{event.name}' already exists.")
        self.register_event(event)

    def get_event(self, event_name: str) -> TestEvent | None:
        event_data = self.data["events"].get(event_name)
        if event_data:
            return TestEvent(**event_data)
        return None

    def get_event_or_raise(self, event_name: str) -> TestEvent:
        event_data = self.data["events"].get(event_name)
        if event_data:
            return TestEvent(**event_data)
        raise Exception(f"No event: {event_name}")


def asyncit(func: Callable, *args, **kwargs):
    print("Got kwargs", kwargs.keys())
    """Wrap a non-async function to run in the background as an asyncio task."""

    async def async_func(*args, **kwargs):
        # Run the function in a background thread using asyncio.to_thread
        try:
            return await asyncio.to_thread(func, *args, **kwargs)
        except Exception as e:
            print(f"An error occurred in asyncit: {e}")

    loop = get_or_create_event_loop()

    # Schedule the async function to run as a background task
    return loop.create_task(async_func(*args, **kwargs))


class AsyncWaitForEvent:
    def __init__(
        self,
        event_manager,
        event_name: str,
        retry_secs: int = 1,
        timeout_secs: int = None,
    ):
        self.event_manager = event_manager
        self.event_name = event_name
        self.retry_secs = retry_secs
        self.timeout_secs = timeout_secs
        self.event_occurred = asyncio.Event()

    async def _event_waiter(self):
        """Internal method that runs asynchronously to wait for the event."""
        elapsed_time = 0
        while not self.event_manager.get_event(self.event_name):
            await asyncio.sleep(self.retry_secs)
            elapsed_time += self.retry_secs
            if self.timeout_secs is not None and elapsed_time >= self.timeout_secs:
                break
        self.event_occurred.set()

    async def __aenter__(self):
        """Starts the event waiter task and waits for the event to occur before returning."""
        self.waiter_task = get_or_create_event_loop().create_task(self._event_waiter())
        await self.event_occurred.wait()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Ensure the event waiter task completes before exiting the context."""
        await self.waiter_task


class WaitForEvent:
    def __init__(
        self,
        event_manager,
        event_name: str,
        retry_secs: int = 1,
        timeout_secs: int = None,
    ):
        self.event_manager = event_manager
        self.event_name = event_name
        self.retry_secs = retry_secs
        self.timeout_secs = timeout_secs
        self.event_occurred = ThreadingEvent()

    def _event_waiter(self):
        """Internal method that runs synchronously to wait for the event."""
        elapsed_time = 0
        while not self.event_manager.get_event(self.event_name):
            time.sleep(self.retry_secs)
            elapsed_time += self.retry_secs
            if self.timeout_secs is not None and elapsed_time >= self.timeout_secs:
                break
        self.event_occurred.set()

    def __enter__(self):
        """Starts the event waiter and waits for the event to occur."""
        self._event_waiter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Nothing specific to do for synchronous exit."""
        pass
