import asyncio
import queue

from queue import Empty
from time import time

from dataclasses import dataclass
from dataclasses import field


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


class TimeoutQueue(queue.Queue):
    class NotFinished(Exception):
        pass

    def join_with_timeout(self, timeout):
        self.all_tasks_done.acquire()
        try:
            endtime = time() + timeout
            while self.unfinished_tasks:
                remaining = endtime - time()
                if remaining <= 0.0:
                    raise TimeoutQueue.NotFinished
                self.all_tasks_done.wait(remaining)
        except TimeoutQueue.NotFinished:
            pass
        finally:
            self.all_tasks_done.release()


class ToyWorker:
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
        return str(self.name) + str(ToyWorker.counter)

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
        ToyWorker.counter += 1

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


class SynchronousWorker(ToyWorker):
    pass


class AsynchronousWorker(ToyWorker):
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


class ThreadedSynchronousWorker(ToyWorker):
    def __init__(self, name: str, actions: list = [], all_workers: dict = {}):
        super().__init__(name, actions, all_workers)

        # Shared queues for communication between workers
        self.incoming = {}
        self.outgoing = {}

    def receive_msgs(self):
        for queue in self.incoming.values():
            try:
                message = queue.get(timeout=self.timeout)
                self._execute_action(message.to_action())
                # Try to execute anything that is now unlocked by received values
                self.execute()
            except Empty:
                # There was no message
                pass

    def send_msg(self, message: ToyMessage):
        queue = self.outgoing[message.destination]

        # Send a message and wait for it to be processed
        queue.put(message)
        queue.join_with_timeout(self.timeout)

    def execute(self, timeout=None):
        if timeout:
            self.timeout = timeout

        while not self.finished:
            action = self.actions[self.next_action_index]

            # Run the command if dependencies are satisfied
            if all(dep in self.store.keys() for dep in action.dependencies):
                self.next_action_index += 1
                self._execute_action(action)

            # Check for incoming messages that might unblock the next action
            self.receive_msgs()

    @staticmethod
    def introduce(*args):
        # Build map of all the workers
        workers = {}
        for worker in args:
            workers[worker.name] = worker

        # Introduce them to each other
        for worker in args:
            worker.known_workers = workers

        # Set up incoming and outgoing queues
        for worker in args:
            for peer_name in worker.known_workers:
                shared_queue = TimeoutQueue()
                worker.incoming[peer_name] = shared_queue
                workers[peer_name].outgoing[worker.name] = shared_queue
