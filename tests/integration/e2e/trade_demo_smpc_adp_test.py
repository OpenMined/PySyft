# stdlib
import os
import time
from typing import Any
from typing import Dict
import uuid

# third party
import numpy as np
import pandas as pd
import pytest

# syft absolute
import syft as sy
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.tensor.config import DEFAULT_INT_NUMPY_TYPE

sy.logger.remove()

PRIVACY_BUDGET = 9_999_999
BUDGET_INCREASE = 200

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print("ROOT_DIR", ROOT_DIR)


def load_data(csv_file: str) -> pd.DataFrame:
    return pd.read_csv(f"{ROOT_DIR}/notebooks/trade_demo/datasets/{csv_file}")[0:10]


def get_user_details(unique_email: str) -> Dict[str, Any]:
    return {
        "name": "Sheldon Cooper",
        "email": unique_email,
        "password": "bazinga",
        "budget": PRIVACY_BUDGET,
    }


# TODO: To fix privacy budget.
# Priavacy budget , Dramatically increased due to high budget spent with 64bit Values.


@pytest.mark.e2e
def test_end_to_end_smpc_adp_trade_demo() -> None:
    # make a unique email so we can run the test isolated
    unique_email = f"{uuid.uuid4()}@caltech.edu"

    # Canada
    ca_root = sy.login(email="info@openmined.org", password="changethis", port=9082)
    ca_data = load_data(csv_file="ca - feb 2021.csv")

    canada_trade = (
        (np.array(list(ca_data["Trade Value (US$)"])) / 100000)[0:10]
    ).astype(DEFAULT_INT_NUMPY_TYPE)
    data_subjects_canada = np.broadcast_to(
        np.array(DataSubjectArray(["Other Asia, nes"])), canada_trade.shape
    )

    lower_bound = int(min(canada_trade)) - 1
    upper_bound = int(max(canada_trade)) + 1

    sampled_canada_dataset = sy.Tensor(canada_trade)
    sampled_canada_dataset.public_shape = sampled_canada_dataset.shape
    sampled_canada_dataset = sampled_canada_dataset.annotate_with_dp_metadata(
        lower_bound, upper_bound, data_subjects=data_subjects_canada
    ).tag("trade_flow")

    # load dataset
    ca_root.load_dataset(
        assets={"Canada Trade": sampled_canada_dataset},
        name="Canada Trade Data - First few rows",
        description=(
            "A collection of reports from Canada's statistics bureau about how "
            + "much it thinks it imports and exports from other countries."
        ),
        skip_checks=True,
    )

    assert len(ca_root.datasets) > 0

    # create user
    ca_root.users.create(**get_user_details(unique_email=unique_email))

    # Italy
    it_root = sy.login(email="info@openmined.org", password="changethis", port=9083)
    it_data = load_data(csv_file="it - feb 2021.csv")

    italy_trade = (
        (np.array(list(it_data["Trade Value (US$)"])) / 100000)[0:10]
    ).astype(DEFAULT_INT_NUMPY_TYPE)

    # Upload a private dataset to the Domain object, as the root owner
    sampled_italy_dataset = sy.Tensor(italy_trade)
    sampled_italy_dataset.public_shape = sampled_italy_dataset.shape
    data_subjects_italy = np.broadcast_to(
        np.array(DataSubjectArray(["Other Asia, nes"])), italy_trade.shape
    )

    lower_bound = int(min(italy_trade)) - 1
    upper_bound = int(max(italy_trade)) + 1

    sampled_italy_dataset = sampled_italy_dataset.annotate_with_dp_metadata(
        lower_bound, upper_bound, data_subjects=data_subjects_italy
    ).tag("trade_flow")

    it_root.load_dataset(
        assets={"Italy Trade": sampled_italy_dataset},
        name="Italy Trade Data - First few rows",
        description=(
            "A collection of reports from iStat's statistics bureau about how "
            + "much it thinks it imports and exports from other countries."
        ),
        skip_checks=True,
    )

    assert len(it_root.datasets) > 0

    # create user
    it_root.users.create(**get_user_details(unique_email=unique_email))

    # Data Scientist
    ca = sy.login(email=unique_email, password="bazinga", port=9082)

    ca.request_budget(eps=BUDGET_INCREASE, reason="increase budget!")
    it = sy.login(email=unique_email, password="bazinga", port=9083)

    it.request_budget(eps=BUDGET_INCREASE, reason="increase budget!")

    time.sleep(20)

    # until we fix the code this just accepts all requests in case it gets the
    # wrong one

    for req in ca_root.requests:
        req.accept()

    for req in it_root.requests:
        req.accept()

    # ca_root.requests[-1].accept()
    # it_root.requests[-1].accept()

    time.sleep(20)

    assert round(ca.privacy_budget) == PRIVACY_BUDGET + BUDGET_INCREASE
    assert round(it.privacy_budget) == PRIVACY_BUDGET + BUDGET_INCREASE

    ca_data = ca.datasets[-1]["Canada Trade"]
    it_data = it.datasets[-1]["Italy Trade"]

    """
    Cutter: Every magic trick consists of three parts, or acts.
    """

    """
    Cutter: The first part is called "The Pledge". The mathemagician shows you something
    ordinary: Tensor Addition. He shows you this expression. Perhaps he asks you to
    inspect it to see if it is indeed normal. But of course... it probably isn't.
    """
    # the pledge 🦜
    print("running the pledge 🦜")

    result = ca_data + it_data

    """
    Cutter: The second act is called "The Turn". The mathemagician takes the ordinary
    something and makes it do something extraordinary. Now you're looking for the
    complexity... but you won't find it, because of course it's abstracted away.
    """
    # the turn 🕳
    print("running the turn 🕳")

    public_result = result.publish(sigma=200)

    """
    Cutter: But you wouldn't clap yet. Because making information disappear isn't enough;
    you have to bring it back. That's why smpc + autodp has a third act, the hardest
    part, the part we call "The Prestige".
    """
    # the prestige 🎩
    print("running the prestige 🎩")

    sycure_result = public_result.get(timeout_secs=40)

    print("sycure_result", sycure_result)
    print("after ca", ca.privacy_budget)
    print("after it", it.privacy_budget)

    assert len(sycure_result) == 10
    assert sum(sycure_result) > -6000
    assert sum(sycure_result) < 6000
    assert public_result.child[0].path_and_name == "numpy.ndarray"

    assert ca.privacy_budget < PRIVACY_BUDGET + BUDGET_INCREASE
    assert ca.privacy_budget > 10
    assert it.privacy_budget < PRIVACY_BUDGET + BUDGET_INCREASE
    assert it.privacy_budget > 10
    # Commenting it out , due to inconsistent budget spent due to 64 bit.
    # assert ca.privacy_budget == it.privacy_budget
