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
            "vinal": "PASSED",
            "yangyuqiao": "PASSED",
            "callis": "PASSED",
            "rodrigo": "PASSED",
            "kanak": "PASSED",
            "theresa": "PASSED",
            "amdjed": "PASSED",
        }
    }
    return data[cohort]


def test_trial_of_skill() -> None:
    assert get_padawans("R2Q4")["skywalker"] == "PASSED"
    assert get_padawans("R2Q4")["saffron"] == "PASSED"
    assert get_padawans("R2Q4")["vinal"] == "PASSED"
    assert get_padawans("R2Q4")["yash"] == "PASSED"
    assert get_padawans("R2Q4")["yangyuqiao"] == "PASSED"
    assert get_padawans("R2Q4")["callis"] == "PASSED"
    assert get_padawans("R2Q4")["rodrigo"] == "PASSED"
    assert get_padawans("R2Q4")["kanak"] == "PASSED"
    assert get_padawans("R2Q4")["theresa"] == "PASSED"
    assert get_padawans("R2Q4")["amdjed"] == "PASSED"
    assert len(get_padawans("R2Q4")) > 1
