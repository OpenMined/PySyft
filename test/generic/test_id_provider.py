import unittest.mock as mock

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
