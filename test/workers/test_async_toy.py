import pytest
import asyncio

from syft.workers.async_toy import ToyWorker
from syft.workers.async_toy import SynchronousWorker
from syft.workers.async_toy import AsynchronousWorker
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
