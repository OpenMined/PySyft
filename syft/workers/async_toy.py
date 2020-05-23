from dataclasses import dataclass
from dataclasses import field

import asyncio


@dataclass
class ToyAction:
    command: str
    value: str = field(default_factory=str)
    dependencies: list = field(default_factory=list)


@dataclass
class ToyMessage:
    destination: str
    command: str
    value: str

    def to_action(self):
        return ToyAction(dependencies=[], command=self.command, value=self.value)


class AbstractToyWorker:
    counter = 0

    def __init__(self, name: str, actions: list = [], all_workers: dict = {}):
        self.name = name
        self.actions = actions
        self.known_workers = all_workers
        self.store = {}
        self.next_action_index = 0
        self.executed = []

    @property
    def _name_counter(self):
        return str(self.name) + str(AbstractToyWorker.counter)

    def receive_msg(self, message: ToyMessage):
        self._execute_action(message.to_action())
        # Try to execute anything that is now unlocked by received values
        self.execute()

    def send_msg(self, message: ToyMessage):
        self.known_workers[message.destination].receive_msg(message)

    def execute(self):
        while not self.finished:
            action = self.actions[self.next_action_index]

            # Check dependencies and return if not satisfied
            for dep in action.dependencies:
                if dep not in self.store.keys():
                    return

            # Run the command if dependencies are satisfied
            self.next_action_index += 1
            self._execute_action(action)

    def _execute_action(self, action):
        # Record the action being executed
        self.executed.append(action)

        # Set a flag in the store and execute any further actions now unlocked
        if action.command == "set" or action.command == "receive":
            self.store[action.value] = True
        # Send a message that sets a flag on a remote worker
        elif action.command == "send":
            msg = ToyMessage(destination=action.value, command="receive", value=self._name_counter)
            return self.send_msg(msg)
        # Do a local "computation"
        elif action.command == "compute":
            self.store[self._name_counter] = True
        else:
            raise ValueError("action.command value not supported.")

        # Increment the global counter of actions executed
        AbstractToyWorker.counter += 1

        return None

    @property
    def finished(self):
        return self.next_action_index >= len(self.actions)

    @classmethod
    def reset_counter(cls):
        cls.counter = 0

    @staticmethod
    def introduce(*args):
        # Build map of all the workers
        workers = {}
        for worker in args:
            workers[worker.name] = worker

        # Introduce them to each other
        for worker in args:
            worker.known_workers = workers


class SynchronousWorker(AbstractToyWorker):
    pass


class AsynchronousWorker(AbstractToyWorker):
    async def receive_msg(self, message: ToyMessage):
        await self._execute_action(message.to_action())
        # Try to execute anything that is now unlocked by received values
        await self.execute()

    async def send_msg(self, message: ToyMessage):
        await self.known_workers[message.destination].receive_msg(message)

    async def execute(self):
        while not self.finished:
            action = self.actions[self.next_action_index]

            # Check dependencies and return if not satisfied
            for dep in action.dependencies:
                if dep not in self.store.keys():
                    return

            # Run the command if dependencies are satisfied
            self.next_action_index += 1
            await self._execute_action(action)

    async def _execute_action(self, action):
        awaitable = super()._execute_action(action)
        if awaitable:
            await awaitable
