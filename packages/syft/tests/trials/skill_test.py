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
            "vinal": "PASSED",
            "yangyuqiao": "PASSED",
            "callis": "PASSED",
            "rodrigo": "PASSED",
            "kanak": "PASSED",
            "Simran": "PASSED",
            "theresa": "PASSED",
            "amdjed": "PASSED",
            "Osam": "PASSED",
            "Mikaela": "PASSED",
            "Nilansh": "PASSED",
            "Anna": "PASSED",
        }
    }
    return data[cohort]


def test_trial_of_skill() -> None:
    assert get_padawans("R2Q4")["skywalker"] == "PASSED"
    assert get_padawans("R2Q4")["saffron"] == "PASSED"
    assert get_padawans("R2Q4")["vinal"] == "PASSED"
    assert get_padawans("R2Q4")["yash"] == "PASSED"
    assert get_padawans("R2Q4")["omar"] == "PASSED"
    assert get_padawans("R2Q4")["yangyuqiao"] == "PASSED"
    assert get_padawans("R2Q4")["callis"] == "PASSED"
    assert get_padawans("R2Q4")["rodrigo"] == "PASSED"
    assert get_padawans("R2Q4")["kanak"] == "PASSED"
    assert get_padawans("R2Q4")["Simran"] == "PASSED"
    assert get_padawans("R2Q4")["theresa"] == "PASSED"
    assert get_padawans("R2Q4")["amdjed"] == "PASSED"
    assert get_padawans("R2Q4")["Osam"] == "PASSED"
    assert get_padawans("R2Q4")["Mikaela"] == "PASSED"
    assert get_padawans("R2Q4")["Nilansh"] == "PASSED"
    assert get_padawans("R2Q4")["Anna"] == "PASSED"
    assert len(get_padawans("R2Q4")) > 1
