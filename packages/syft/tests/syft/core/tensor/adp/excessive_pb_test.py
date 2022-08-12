# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# syft absolute
from syft.core.adp.data_subject_ledger import DataSubjectLedger
from syft.core.adp.data_subject_list import DataSubjectArray
from syft.core.adp.ledger_store import DictLedgerStore
from syft.core.tensor.autodp.gamma_tensor import GammaTensor
from syft.core.tensor.lazy_repeat_array import lazyrepeatarray


@pytest.fixture
def reference_tensor():
    return GammaTensor(
        child=np.random.random((10, 10)),
        data_subjects=DataSubjectArray.from_objs(np.random.choice([0, 1], (10, 10))),
        min_vals=lazyrepeatarray(0, (10, 10)),
        max_vals=lazyrepeatarray(1, (10, 10)),
    )


def test_nan_pb(reference_tensor):
    tensor = reference_tensor + np.nan

    ledger_store = DictLedgerStore()
    user_key = b"1231"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)

    def get_budget_for_user(*args: Any, **kwargs: Any) -> float:
        return 999999

    def deduct_epsilon_for_user(*args: Any, **kwargs: Any) -> bool:
        return True

    with pytest.raises(Exception):
        _ = tensor.publish(
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            ledger=ledger,
            sigma=10,
        )


def test_inf_pb(reference_tensor):
    tensor = reference_tensor

    ledger_store = DictLedgerStore()
    user_key = b"1231"
    ledger = DataSubjectLedger.get_or_create(store=ledger_store, user_key=user_key)

    def get_budget_for_user(*args: Any, **kwargs: Any) -> float:
        return 999999

    def deduct_epsilon_for_user(*args: Any, **kwargs: Any) -> bool:
        return True

    with pytest.raises(Exception):
        _ = tensor.publish(
            get_budget_for_user=get_budget_for_user,
            deduct_epsilon_for_user=deduct_epsilon_for_user,
            ledger=ledger,
            sigma=1e-20,
        )
