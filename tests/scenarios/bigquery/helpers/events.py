# stdlib
import asyncio
from dataclasses import dataclass
from functools import wraps
import inspect
import json
import os
from threading import Lock
import time

# third party
import unsync as unsync_lib

__all__ = ["Event", "EventManager", "Scenario", "unsync"]


class Event:
    USER_ADMIN_CREATED = "user_admin_created"
    USERS_CREATED = "users_created"
    DATASET_UPLOADED = "dataset_uploaded"
    DATASET_MOCK_READABLE = "dataset_mock_readable"
    PREBUILT_WORKER_IMAGE_BIGQUERY_CREATED = "prebuilt_worker_image_bigquery_created"
    EXTERNAL_REGISTRY_BIGQUERY_CREATED = "external_registry_bigquery_created"
    WORKER_POOL_CREATED = "worker_pool_created"
    WORKER_POOL_SCALED = "worker_pool_scaled"
    ALLOW_GUEST_SIGNUP_ENABLED = "allow_guest_signup_enabled"
    ALLOW_GUEST_SIGNUP_DISABLED = "allow_guest_signup_disabled"
    USERS_CREATED_CHECKED = "users_created_checked"
    SCHEMA_ENDPOINT_CREATED = "schema_endpoint_created"
    SUBMIT_QUERY_ENDPOINT_CREATED = "submit_query_endpoint_created"
    SUBMIT_QUERY_ENDPOINT_CONFIGURED = "submit_query_endpoint_configured"
    USERS_CAN_QUERY_MOCK = "users_can_query_mock"
    USERS_CAN_SUBMIT_QUERY = "users_can_submit_query"
    ADMIN_APPROVED_REQUEST = "admin_approved_request"
    ADMIN_APPROVED_FIRST_REQUEST = "admin_approved_first_request"
    USERS_CAN_GET_APPROVED_RESULT = "users_can_get_approved_result"
    USERS_QUERY_NOT_READY = "users_query_not_ready"
    QUERY_ENDPOINT_CREATED = "query_endpoint_created"
    QUERY_ENDPOINT_CONFIGURED = "query_endpoint_configured"

    ADMIN_LOW_SIDE_WORKFLOW_COMPLETED = "admin_low_side_workflow_completed"
    ADMIN_HIGH_SIDE_WORKFLOW_COMPLETED = "admin_high_side_workflow_completed"
    ADMIN_SYNC_HIGH_TO_LOW = "admin_sync_high_to_low"
    ADMIN_SYNC_LOW_TO_HIGH = "admin_sync_low_to_high"

    EXCEPTION_OCCURRED = "exception_occurred"


@dataclass
class Scenario:
    name: str
    events: list[str]

    def add_event(self, event: str):
        self.events.append(event)


class EventManager:
    def __init__(
        self,
        test_name: str | None = None,
        test_dir: str | None = None,
        reset: bool = True,
    ):
        self.start_time = time.time()
        self.event_file = self._get_event_file(test_name, test_dir)
        self.lock = Lock()
        self._ensure_file_exists()
        self.scenarios = {}
        if reset:
            self.clear_events()

    def add_scenario(self, scenario: Scenario):
        with self.lock:
            with open(self.event_file, "r+") as f:
                events = json.load(f)
                for event in scenario.events:
                    if event not in events:
                        events[event] = None
                self.scenarios[scenario.name] = scenario.events
                f.seek(0)
                json.dump(events, f)
                f.truncate()

    def wait_scenario(
        self, scenario_name: str, timeout: float = 15.0, show: bool = True
    ) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.scenario_completed(scenario_name):
                return True
            if show:
                time_left = timeout - (time.time() - start_time)
                print(f"wait_for_scenario: {scenario_name}. Time left: {time_left}")

            time.sleep(1)
        return False

    async def await_scenario(
        self, scenario_name: str, timeout: float = 15.0, show: bool = True
    ) -> bool:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.scenario_completed(scenario_name):
                return True
            if show:
                time_left = timeout - (time.time() - start_time)
                print(
                    f"async await_for_scenario: {scenario_name}. Time left: {time_left}"
                )
            await asyncio.sleep(1)
        return False

    def scenario_completed(self, scenario_name: str) -> bool:
        with self.lock:
            with open(self.event_file) as f:
                events = json.load(f)
                scenario_events = self.scenarios.get(scenario_name, [])
                incomplete_events = [
                    event for event in scenario_events if events.get(event) is None
                ]
                if Event.EXCEPTION_OCCURRED in events:
                    msg = f"Scenario '{scenario_name}' failed due to an exception. Missing events: {incomplete_events}"
                    raise Exception(msg)

                if incomplete_events:
                    print(
                        f"Scenario '{scenario_name}' is incomplete. Missing events: {incomplete_events}"
                    )
                    return False
                return True

    def _get_event_file(
        self, test_name: str | None = None, test_dir: str | None = None
    ):
        # Get the calling test function's name
        if not test_name:
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back
            while caller_frame:
                if caller_frame.f_code.co_name.startswith("test_"):
                    test_name = caller_frame.f_code.co_name
                    break
                caller_frame = caller_frame.f_back
            else:
                test_name = "unknown_test"

        # Get the directory of the calling test file
        if not test_dir:
            current_frame = inspect.currentframe()
            caller_frame = current_frame.f_back
            caller_file = inspect.getfile(caller_frame)
            test_dir = os.path.dirname(os.path.abspath(caller_file))

        # Create a unique filename for this test
        return os.path.join(test_dir, f"{test_name}_events.json.events")

    def _ensure_file_exists(self):
        if not os.path.exists(self.event_file):
            with open(self.event_file, "w") as f:
                json.dump({}, f)

    def register(self, event_name: str):
        with self.lock:
            with open(self.event_file, "r+") as f:
                now = time.time()
                events = json.load(f)
                events[event_name] = now
                f.seek(0)
                json.dump(events, f)
                f.truncate()
        print(f"> Event: {event_name} occured at: {now}")

    def wait_for(
        self,
        event_name: str | list[str] | tuple[str],
        timeout: float = 15.0,
        show: bool = True,
    ) -> bool:
        event_names = event_name
        if isinstance(event_names, str):
            event_names = [event_names]

        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(self.happened(event_name) for event_name in event_names):
                return True
            if show:
                time_left = timeout - (time.time() - start_time)
                print(f"wait_for: {event_names}. Time left: {time_left}")

            time.sleep(1)
        return False

    async def await_for(
        self,
        event_name: str | list[str] | tuple[str],
        timeout: float = 15.0,
        show: bool = True,
    ) -> bool:
        event_names = event_name
        if isinstance(event_names, str):
            event_names = [event_names]

        start_time = time.time()
        while time.time() - start_time < timeout:
            if all(self.happened(event_name) for event_name in event_names):
                return True
            if show:
                time_left = timeout - (time.time() - start_time)
                print(f"async await_for: {event_names}. Time left: {time_left}")
            await asyncio.sleep(5)
        return False

    def happened(self, event_name: str) -> bool:
        try:
            with self.lock:
                with open(self.event_file) as f:
                    events = json.load(f)
                    if event_name in events:
                        return events[event_name]
        except Exception as e:
            print("e", e)
        return False

    def get_event_time(self, event_name: str) -> float | None:
        with self.lock:
            with open(self.event_file) as f:
                events = json.load(f)
                return events.get(event_name)

    def clear_events(self):
        with self.lock:
            with open(self.event_file, "w") as f:
                json.dump({}, f)

    @unsync_lib.unsync
    async def monitor(self, period: float = 2):
        while True:
            await asyncio.sleep(period)
            self.flush_monitor()

    def flush_monitor(self):
        with self.lock:
            with open(self.event_file) as f:
                events = json.load(f)
                if not events:
                    return
                for event, timestamp in sorted(events.items(), key=lambda x: x[1]):
                    if timestamp:
                        now = time.time()
                        time_since_start = now - timestamp
                        print(
                            f"Event: {event} happened {time_since_start:.2f} seconds ago"
                        )
                    else:
                        print(
                            f"Event: {event} is registered but has not happened yet. Pending..."
                        )

    def __del__(self):
        # Clean up the file when the EventManager is destroyed
        # if os.path.exists(self.event_file):
        #     os.remove(self.event_file)
        pass


def unsync_guard():
    "Make sure we exit early if an exception occurs"

    def decorator(func):
        @wraps(func)
        @unsync_lib.unsync
        async def wrapper(*args, **kwargs):
            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                print(f"Exception occurred: {e}")
                for arg in args:
                    if isinstance(arg, EventManager):
                        print("Registering exception event")
                        arg.register(Event.EXCEPTION_OCCURRED)
                        break
                raise

        return wrapper

    return decorator


unsync = unsync_guard()
