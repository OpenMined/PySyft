# stdlib
from typing import Any

# third party
import numpy as np
import pytest
import torch

# syft absolute
from syft import UID
from syft.service.action.action_object import ActionObject
from syft.service.user.user import User
from syft.util.table import TABLE_INDEX_KEY
from syft.util.table import prepare_table_data


def table_displayed(obj_to_check: Any) -> bool:
    return "Tabulator" in obj_to_check._repr_html_()


def no_html_repr_displayed(obj_to_check: Any) -> bool:
    return obj_to_check._repr_html_() is None


def obj_repr_displayed(obj_to_check: Any) -> bool:
    return obj_to_check._repr_html_() == obj_to_check.__repr__()


def table_test_cases() -> list[tuple[list, str | None]]:
    ao_1 = ActionObject.from_obj(10.0)
    ao_2 = ActionObject.from_obj(20.0)
    np_ao = ActionObject.from_obj(np.array([10, 20]))
    torch_ao = ActionObject.from_obj(torch.tensor([10, 20]))
    user_1 = User(email="x@y.z")
    user_2 = User(email="a@b.c")

    # Makes table
    homogenous_ao = ([ao_1, ao_2], table_displayed)
    non_homogenous_same_repr = ([ao_1, ao_2, np_ao], table_displayed)
    homogenous_user = ([user_1, user_2], table_displayed)
    empty_list = ([], obj_repr_displayed)
    non_syft_objs = ([1, 2.0, 3, 4], no_html_repr_displayed)

    # Doesn't make table
    non_homogenous_different_repr = (
        [ao_1, ao_2, user_1, user_2],
        no_html_repr_displayed,
    )
    non_syft_obj_1 = ([1, ao_1, ao_2], no_html_repr_displayed)
    non_syft_obj_2 = ([ao_1, ao_2, 1], no_html_repr_displayed)
    torch_type_obj = (
        [type(torch_ao.syft_action_data), 1.0, UID()],
        no_html_repr_displayed,
    )
    return [
        homogenous_ao,
        non_homogenous_same_repr,
        homogenous_user,
        empty_list,
        non_syft_objs,
        non_homogenous_different_repr,
        non_syft_obj_1,
        non_syft_obj_2,
        torch_type_obj,
    ]


@pytest.mark.parametrize("test_case", table_test_cases())
def test_list_dict_repr_html(test_case):
    obj, validation_func = test_case

    assert validation_func(obj)
    assert validation_func(dict(enumerate(obj)))
    assert validation_func(set(obj))
    assert validation_func(tuple(obj))


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
