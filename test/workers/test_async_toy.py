import asyncio
import pytest
import threading

from syft.workers.async_toy import ToyWorker
from syft.workers.async_toy import AsynchronousWorker
from syft.workers.async_toy import SynchronousWorker
from syft.workers.async_toy import ThreadedSynchronousWorker
from syft.workers.async_toy import ToyAction
from syft.workers.async_toy import ToyMessage


# The "set" action sets a boolean flag in the worker's object store. This is a stand-in for storing
# a tensor and allows a similar style of checking for the availability of action dependencies.
def test_set():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    action = ToyAction("set", "test")

    alice._execute_action(action)
    assert alice.store["test"] is True


# The "compute" action concatenates the worker's name with the current global counter of actions
# executed and uses that value to set a flag in the worker's object store. This simulates performing
# tensor operations and storing the output for later use.
def test_compute():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    action = ToyAction("compute", "test")

    alice._execute_action(action)

    assert alice.store == {"alice0": True}


# The "send" action computes the same "name-counter" value as the "compute" action, but instead of
# setting the corresponding flag in the local object store, it sends the value to a remote worker
# and instructs the remote worker to set that flag in its object store. This simulates sending a
# tensor value to a remote worker that may be used as an input to the remote worker's future
# actions.
def test_send():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    bob = ToyWorker(name="bob")
    ToyWorker.introduce(alice, bob)

    action = ToyAction("send", "bob")

    alice._execute_action(action)
    assert bob.store["alice0"] is True


# Unsupported actions should raise an exception
def test_unknown_action():
    ToyWorker.reset_counter()

    alice = ToyWorker(name="alice")
    action = ToyAction("flarble", "garb")

    with pytest.raises(ValueError):
        alice._execute_action(action)


# The global action counter should be incremented every time any worker executes an action.
# Trasmitting a message between workers only counts as one action, although both the sending and
# receiving sides technically execute an action under the hood.
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

    # Send actions increment the counter only for send (and not receive)
    action = ToyAction("send", "bob")
    alice._execute_action(action)

    assert bob.store["alice3"] is True
    assert ToyWorker.counter == 4


# This case is similar to VirtualWorkers taking turns executing as many actions as they can in a
# round-robin loop. In this case, the call stack gets rather deeply nested, since the workers are
# sending message by directly calling each other's receive methods, but it works fine, even when
# Alice sends a value to Bob that is immediately used to compute a value that is then sent back to
# Alice. There's no deadlock, because only one worker can be executing something at a time.
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


# This case is almost identical to the test above, except now the workers have been converted to use
# async/await. Although you might expect that this means they are executing actions concurrently,
# they're actually still taking turns. When we `await alice.execute()`, that blocks until the method
# call has completed. This code is asynchronous but not concurrent, and does not deadlock and still
# works.
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


# Here we update the way the `worker.execute()` is invoked, so that the workers execute actions
# concurrently, in addition to asynchronously. There's still no deadlock here, because the
# coroutines used by async/await are just fancy single-threaded turn-taking that's built into
# Python.
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


# Now we try to run synchronous blocking code in parallel threads. In order to make this work, the
# workers can't directly call each other's receive methods any more, so they communicate using
# shared thread-safe queues. When Alice drops a message on her outgoing queue to Bob, she waits
# until Bob has cleared the queue before resuming execution. This simulates a blocking network call.

# The first set of timeout parameters should pass, because the blocking communication methods will
# wait for 1 second before timing out, and the whole execution is 10 seconds long, leaving plenty of
# time for retries after timeouts happen.

# The second set of timeout parameters should fail, because the blocking communication methods will
# wait for 10 seconds before timing out, and the whole execution is only 5 seconds long, creating a
# deadlock that doesn't resolve before the execution is terminated. (If there were no timeouts, this
# would block forever and the test would hang instead of failing.)

# Even though the first test passes, it's worth noting that both sets of parameters cause this test
# to execute incredibly slowly compared to the other tests. This is a major disadvantage of running
# synchronous blocking code in parallel with threads, as compared to running asynchronous
# non-blocking code in a single thread. Intuitively, it seems like it should be faster to run
# multiple threads, but it turns out to be way, way slower than using a single thread efficiently.


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
