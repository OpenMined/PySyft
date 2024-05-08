# third party
import numpy as np
import pytest

# syft absolute
from syft.service.action.action_object import ActionObject
from syft.service.user.user import User
from syft.util.table import TABLE_INDEX_KEY
from syft.util.table import prepare_table_data


def table_test_cases() -> list[tuple[list, str | None]]:
    ao_1 = ActionObject.from_obj(10.0)
    ao_2 = ActionObject.from_obj(20.0)
    np_ao = ActionObject.from_obj(np.array([10, 20]))
    user_1 = User(email="x@y.z")
    user_2 = User(email="a@b.c")

    # Makes table
    homogenous_ao = ([ao_1, ao_2], True)
    non_homogenous_same_repr = ([ao_1, ao_2, np_ao], True)
    homogenous_user = ([user_1, user_2], True)
    # TODO techdebt: makes table because syft misuses _html_repr_
    empty_list = ([], True)

    # Doesn't make table
    non_homogenous_different_repr = ([ao_1, ao_2, user_1, user_2], False)
    non_syft_obj_1 = ([1, ao_1, ao_2], False)
    non_syft_obj_2 = ([ao_1, ao_2, 1], False)

    return [
        homogenous_ao,
        non_homogenous_same_repr,
        homogenous_user,
        empty_list,
        non_homogenous_different_repr,
        non_syft_obj_1,
        non_syft_obj_2,
    ]


@pytest.mark.parametrize("test_case", table_test_cases())
def test_list_dict_repr_html(test_case):
    obj, expected = test_case

    assert (obj._repr_html_() is not None) == expected
    assert (dict(enumerate(obj))._repr_html_() is not None) == expected
    assert (set(obj)._repr_html_() is not None) == expected
    assert (tuple(obj)._repr_html_() is not None) == expected


def test_sort_table_rows():
    emails = [
        "x@y.z",
        "a@b.c",
        "c@d.e",
    ]
    sorted_order = [1, 2, 0]
    users = [User(email=email) for email in emails]

    table_data, _ = prepare_table_data(users)

    # No sorting
    table_emails = [row["email"] for row in table_data]
    table_indices = [row[TABLE_INDEX_KEY] for row in table_data]
    assert table_emails == emails
    assert table_indices == list(range(len(emails)))

    # Sort by email
    User.__table_sort_attr__ = "email"
    table_data_sorted, _ = prepare_table_data(users)
    table_emails_sorted = [row["email"] for row in table_data_sorted]
    table_indices_sorted = [row[TABLE_INDEX_KEY] for row in table_data_sorted]
    assert table_emails_sorted == sorted(emails)
    assert table_indices_sorted == sorted_order
