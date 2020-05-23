import asyncio
import pytest
import threading

from syft.workers.async_toy import ToyWorker
from syft.workers.async_toy import AsynchronousWorker
from syft.workers.async_toy import SynchronousWorker
from syft.workers.async_toy import ThreadedSynchronousWorker
from syft.workers.async_toy import ToyAction
from syft.workers.async_toy import ToyMessage


def test_set():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    action = ToyAction("set", "test")

    alice._execute_action(action)
    assert alice.store["test"] is True


def test_compute():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    action = ToyAction("compute", "test")

    alice._execute_action(action)

    assert alice.store == {"alice0": True}


def test_send():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    bob = ToyWorker(name="bob")
    ToyWorker.introduce(alice, bob)

    action = ToyAction("send", "bob")

    alice._execute_action(action)
    assert bob.store["alice0"] is True


def test_unknown_action():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    action = ToyAction("flarble", "garb")

    with pytest.raises(ValueError):
        alice._execute_action(action)


def test_action_counter():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    bob = ToyWorker(name="bob")
    ToyWorker.introduce(alice, bob)

    # Counter increments by one every time an action is executed
    action = ToyAction("set", "test1")
    alice._execute_action(action)

    assert alice.store["test1"] is True
    assert ToyWorker.counter == 1

    action = ToyAction("set", "test2")
    alice._execute_action(action)

    assert alice.store["test2"] is True
    assert ToyWorker.counter == 2

    # Counter increments regardless of which worker executes the action
    action = ToyAction("set", "test3")
    bob._execute_action(action)

    assert bob.store["test3"] is True
    assert ToyWorker.counter == 3

    # Send actions increment the counter only for send
    action = ToyAction("send", "bob")
    alice._execute_action(action)

    assert bob.store["alice3"] is True
    assert ToyWorker.counter == 4


def test_single_threaded_synchronous_comms():
    ToyWorker.reset_counter()

    alice = SynchronousWorker(name="alice")
    bob = SynchronousWorker(name="bob")
    ToyWorker.introduce(alice, bob)

    # Create a potential deadlock
    alice.actions = [ToyAction("send", "bob"), ToyAction("compute", dependencies=["bob2"])]
    bob.actions = [ToyAction("compute", dependencies=["alice0"]), ToyAction("send", "alice")]

    # Take turns executing synchronously
    while not (alice.finished and bob.finished):
        alice.execute()
        bob.execute()

    assert len(alice.executed) == 3
    assert len(bob.executed) == 3

    assert bob.store["alice0"] is True
    assert bob.store["bob1"] is True
    assert alice.store["bob2"] is True
    assert alice.store["alice3"] is True


@pytest.mark.asyncio
async def test_single_threaded_async_comms():
    ToyWorker.reset_counter()

    alice = AsynchronousWorker(name="alice")
    bob = AsynchronousWorker(name="bob")
    ToyWorker.introduce(alice, bob)

    # Create a potential deadlock
    alice.actions = [ToyAction("send", "bob"), ToyAction("compute", dependencies=["bob2"])]
    bob.actions = [ToyAction("compute", dependencies=["alice0"]), ToyAction("send", "alice")]

    # Take turns executing asynchronously
    while not (alice.finished and bob.finished):
        await alice.execute()
        await bob.execute()

    assert len(alice.executed) == 3
    assert len(bob.executed) == 3

    assert bob.store["alice0"] is True
    assert bob.store["bob1"] is True
    assert alice.store["bob2"] is True
    assert alice.store["alice3"] is True


@pytest.mark.asyncio
async def test_single_threaded_async_comms_concurrent():
    ToyWorker.reset_counter()

    alice = AsynchronousWorker(name="alice")
    bob = AsynchronousWorker(name="bob")
    ToyWorker.introduce(alice, bob)

    # Create a potential deadlock
    alice.actions = [ToyAction("send", "bob"), ToyAction("compute", dependencies=["bob2"])]
    bob.actions = [ToyAction("compute", dependencies=["alice0"]), ToyAction("send", "alice")]

    # Execute asynchronously AND concurrently
    while not (alice.finished and bob.finished):
        await asyncio.gather(alice.execute(), bob.execute())

    assert len(alice.executed) == 3
    assert len(bob.executed) == 3

    assert bob.store["alice0"] is True
    assert bob.store["bob1"] is True
    assert alice.store["bob2"] is True
    assert alice.store["alice3"] is True


# The first set of parameters should pass, because the blocking communication methods will wait for
# 1 second before timing out, and the whole execution is 10 seconds long, leaving plenty of time for
# retries after timeouts happen.

# The second set of parameters should fail, because the blocking communication methods will wait for
# 10 seconds before timing out, and the whole execution is only 5 seconds long, creating a deadlock
# that doesn't resolve before the execution is terminated. (If there were no timeouts, this would
# block forever and the test would hang instead of failing.


@pytest.mark.parametrize("timeout,ratio", [(1, 10), (10, 0.5)])
def test_multi_threaded_parallel_synchronous_comms(timeout, ratio):
    ToyWorker.reset_counter()

    alice = ThreadedSynchronousWorker(name="alice")
    bob = ThreadedSynchronousWorker(name="bob")
    ThreadedSynchronousWorker.introduce(alice, bob)

    # Create a potential deadlock
    alice.actions = [ToyAction("send", "bob"), ToyAction("compute", dependencies=["bob2"])]
    bob.actions = [ToyAction("compute", dependencies=["alice0"]), ToyAction("send", "alice")]

    # Run each worker in a thread
    def run_until_finished(worker, stop_event, timeout):
        while not (worker.finished or stop_event.is_set()):
            worker.execute(timeout)

    stop_event = threading.Event()

    alice_thread = threading.Thread(target=run_until_finished, args=(alice, stop_event, timeout))
    bob_thread = threading.Thread(target=run_until_finished, args=(bob, stop_event, timeout))

    alice_thread.start()
    bob_thread.start()

    # Wait for the threads to finish
    alice_thread.join(timeout=timeout * ratio)
    bob_thread.join(timeout=timeout * ratio)

    stop_event.set()

    assert len(alice.executed) == 3
    assert len(bob.executed) == 3

    assert bob.store["alice0"] is True
    assert bob.store["bob1"] is True
    assert alice.store["bob2"] is True
    assert alice.store["alice3"] is True
