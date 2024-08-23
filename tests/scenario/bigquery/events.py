# third party
import anyio

EVENT_USER_ADMIN_CREATED = "user_admin_created"
EVENT_USERS_CREATED = "users_created"
EVENT_DATASET_UPLOADED = "dataset_uploaded"
EVENT_DATASET_MOCK_READABLE = "dataset_mock_readable"


class EventManager:
    def __init__(self):
        self.events = {}
        self.event_waiters = {}

    def register(self, event_name: str):
        self.events[event_name] = anyio.Event()
        waiters = self.event_waiters.get(event_name, [])
        for waiter in waiters:
            waiter.set()

    async def wait_for(self, event_name: str, timeout: float = 15.0):
        if event_name in self.events:
            return self.events[event_name]

        waiter = anyio.Event()
        self.event_waiters.setdefault(event_name, []).append(waiter)

        try:
            with anyio.move_on_after(timeout) as cancel_scope:
                await waiter.wait()
                if cancel_scope.cancel_called:
                    raise TimeoutError(f"Timeout waiting for event: {event_name}")
            return self.events[event_name]
        finally:
            self.event_waiters[event_name].remove(waiter)
