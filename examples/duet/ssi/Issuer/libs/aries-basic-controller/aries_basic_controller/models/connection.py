import asyncio

class Connection:

    def __init__(self, id, state):
        self.id = id
        self.state = state
        self.future_state = None
        self.future = asyncio.Future()

    async def detect_state_ready(self, future_state):
        if self.state == future_state:
            return True
        else:
            self.future_state = future_state
            await self.future

    def update_state(self, state):
        self.state = state
        if state == self.future_state:
            self.future.set_result(True)

    @property
    def connection_ready(self):
        return self.state == "active"