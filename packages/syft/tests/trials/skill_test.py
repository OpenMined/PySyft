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
            "Jauhar": "PASSED",
            "Nilansh": "PASSED",
            "Anna": "PASSED",
            "Akshay": "PASSED",
            "Zarreen": "PASSED",
            "Mihir": "PASSED",
            "Uche": "PASSED",
        },
        "R3Q1": {
            "skywalker": "PASSED",
            "Zach": "PASSED",
            "Oleksandr": "PASSED",
            "Khoa": "PASSED",
            "Julian": "PASSED",
            "Ajinkya": "PASSED",
            "Hussein": "PASSED",
            "Peter": "PASSED",
            "Vani": "PASSED",
            "Hithem": "PASSED",
        },
        "R4Q2": {
            "skywalker": "PASSED",
            "duarte": "PASSED",
            "alejandrosame": "PASSED",
            "antti": "PASSED",
            "uche": "PASSED",
            "osam_again": "PASSED",
            "paramm": "PASSED",
            "tristan": "PASSED",
        },
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
    assert get_padawans("R2Q4")["Jauhar"] == "PASSED"
    assert get_padawans("R2Q4")["Nilansh"] == "PASSED"
    assert get_padawans("R2Q4")["Anna"] == "PASSED"
    assert get_padawans("R2Q4")["Akshay"] == "PASSED"
    assert get_padawans("R2Q4")["Zarreen"] == "PASSED"
    assert get_padawans("R2Q4")["Mihir"] == "PASSED"
    assert get_padawans("R2Q4")["Uche"] == "PASSED"

    assert len(get_padawans("R2Q4")) == 21

    assert get_padawans("R3Q1")["skywalker"] == "PASSED"
    assert get_padawans("R3Q1")["Zach"] == "PASSED"
    assert get_padawans("R3Q1")["Oleksandr"] == "PASSED"
    assert get_padawans("R3Q1")["Khoa"] == "PASSED"
    assert get_padawans("R3Q1")["Julian"] == "PASSED"
    assert get_padawans("R3Q1")["Ajinkya"] == "PASSED"
    assert get_padawans("R3Q1")["Hussein"] == "PASSED"
    assert get_padawans("R3Q1")["Peter"] == "PASSED"
    assert get_padawans("R3Q1")["Vani"] == "PASSED"
    assert get_padawans("R3Q1")["Hithem"] == "PASSED"

    assert len(get_padawans("R3Q1")) > 1

    assert get_padawans("R4Q2")["skywalker"] == "PASSED"
    assert get_padawans("R4Q2")["alejandrosame"] == "PASSED"
    assert get_padawans("R4Q2")["antti"] == "PASSED"
    assert get_padawans("R4Q2")["uche"] == "PASSED"
    assert get_padawans("R4Q2")["paramm"] == "PASSED"
    assert get_padawans("R4Q2")["osam_again"] == "PASSED"
    assert get_padawans("R4Q2")["tristan"] == "PASSED"
    assert get_padawans("R4Q2")["duarte"] == "PASSED"

    assert len(get_padawans("R4Q2")) == 8
