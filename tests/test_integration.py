import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from grid.clients.keras import KerasClient
from grid.workers.compute import GridCompute
import time
from threading import Thread
import pytest

client = None
compute_id = None


@pytest.fixture
def client():
    client = KerasClient()
    return client


def wait_for_discovery(client):
    ipfs_id = client.id.split(":")
    compute_id = None

    time.sleep(30)

    print(client.stats)
    for stats in client.stats:
        if ipfs_id[1] in stats['id']:
            compute_id = stats['id']

    if compute_id is None:
        time.sleep(15)
    else:
        assert(compute_id is not None)

    # TODO probably shouldn't have to try again
    for stats in client.stats:
        if ipfs_id[1] in stats['id']:
            compute_id = stats['id']

    return compute_id


def test_integration(client):
    compute_id = wait_for_discovery(client)

    assert(compute_id is not None)

    input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    target = np.array([[0], [1], [1], [0]])

    model = Sequential()
    model.add(Dense(8, input_dim=2))
    model.add(Activation('tanh'))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    sgd = SGD(lr=0.1)
    model.compile(loss='binary_crossentropy', optimizer=sgd)

    model, train_spec = client.fit(
        model,
        input,
        target,
        epochs=20,
        log_interval=100,
        preferred_node=f"{compute_id}")
