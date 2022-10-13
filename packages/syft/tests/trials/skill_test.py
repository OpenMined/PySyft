# "Do or do not. There is no try." — Yoda
# stdlib
from typing import Dict


def get_padawans(cohort: str) -> Dict[str, str]:
    # add yourself to the temple trial roster
    data = {
        "R2Q4": {
            "skywalker": "PASSED",
            "saffron": "PASSED",
            "yash": "PASSED",
            "omar": "PASSED",
        }
    }
    return data[cohort]


def test_trial_of_skill() -> None:
    assert get_padawans("R2Q4")["skywalker"] == "PASSED"
    assert get_padawans("R2Q4")["saffron"] == "PASSED"
    assert get_padawans("R2Q4")["yash"] == "PASSED"
    assert get_padawans("R2Q4")["omar"] == "PASSED"
