import unittest.mock as mock
import pytest

from syft import exceptions
from syft.generic import id_provider


def test_pop_no_given_ids(hook):
    provider = id_provider.IdProvider()
    values = [10, 4, 15, 4, 2, 0]

    orig_func = id_provider.create_random_id
    mocked_random_numbers = mock.Mock()
    mocked_random_numbers.side_effect = values
    id_provider.create_random_id = mocked_random_numbers

    val = provider.pop()
    assert val == values[0]

    val = provider.pop()
    assert val == values[1]

    val = provider.pop()
    assert val == values[2]

    # values[3] is skipped, as value already used.

    val = provider.pop()
    assert val == values[4]

    val = provider.pop()
    assert val == values[5]

    id_provider.create_random_id = orig_func


def test_pop_with_given_ids(hook):
    given_ids = [4, 15, 2]
    provider = id_provider.IdProvider(given_ids=given_ids.copy())
    values = [10, 4, 15, 4, 2, 0]

    orig_func = id_provider.create_random_id
    mocked_random_numbers = mock.Mock()
    mocked_random_numbers.side_effect = values
    id_provider.create_random_id = mocked_random_numbers

    val = provider.pop()
    assert val == given_ids[-1]

    val = provider.pop()
    assert val == given_ids[-2]

    val = provider.pop()
    assert val == given_ids[-3]

    val = provider.pop()
    assert val == values[0]

    # values[1, 2, 3, 4] are skipped, as value already used.

    val = provider.pop()
    assert val == values[5]

    id_provider.create_random_id = orig_func


def test_given_ids_side_effect(hook):
    given_ids = [4, 15, 2]
    provider = id_provider.IdProvider(given_ids=given_ids)

    assert len(given_ids) == 3
    provider.pop()

    assert len(given_ids) == 2

    provider.pop()
    assert len(given_ids) == 1

    provider.pop()
    assert len(given_ids) == 0


def test_set_next_ids(hook):
    initial_given_ids = [2, 3]
    provider = id_provider.IdProvider(given_ids=initial_given_ids.copy())

    next_ids = [4, 5]
    provider.set_next_ids(next_ids.copy())

    val = provider.pop()
    assert val == next_ids[-1]
    val = provider.pop()
    assert val == next_ids[-2]

    val = provider.pop()
    assert val == initial_given_ids[-1]
    val = provider.pop()
    assert val == initial_given_ids[-2]


def test_set_next_ids_with_id_checking(hook):
    initial_given_ids = [2, 3]
    provider = id_provider.IdProvider()
    provider.set_next_ids(initial_given_ids.copy(), check_ids=False)

    # generated the initial 3 ids
    provider.pop()
    provider.pop()
    provider.pop()

    next_ids = [1, 2, 5]
    with pytest.raises(exceptions.IdNotUniqueError, match=r"\{2\}"):
        provider.set_next_ids(next_ids.copy(), check_ids=True)

    next_ids = [2, 3, 5]
    with pytest.raises(exceptions.IdNotUniqueError, match=r"\{2, 3\}"):
        provider.set_next_ids(next_ids.copy(), check_ids=True)


def test_start_recording_ids():
    initial_given_ids = [2, 3]
    provider = id_provider.IdProvider(given_ids=initial_given_ids.copy())
    provider.pop()
    provider.start_recording_ids()
    provider.pop()

    ids = provider.get_recorded_ids()
    assert len(ids) == 1
    assert ids[0] == initial_given_ids[-2]


def test_get_recorded_ids():
    initial_given_ids = [2, 3, 4]
    provider = id_provider.IdProvider(given_ids=initial_given_ids.copy())
    provider.pop()
    provider.start_recording_ids()
    provider.pop()

    ids = provider.get_recorded_ids(continue_recording=True)
    assert len(ids) == 1
    assert ids[0] == initial_given_ids[-2]

    provider.pop()

    ids = provider.get_recorded_ids()
    assert len(ids) == 2
    assert ids[0] == initial_given_ids[-2]
    assert ids[1] == initial_given_ids[-3]
