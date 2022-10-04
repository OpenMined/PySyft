# "Do or do not. There is no try." — Yoda


def get_padawans(cohort: str) -> dict[str, str]:
    # add yourself to the temple trial roster

    data = {"R2Q4": {"skywalker": "PASSED", "yangyuqiao": "PASSED"}}

    return data[cohort]


def test_trial_of_skill() -> None:
    assert get_padawans("R2Q4")["skywalker"] == "PASSED"
    assert len(get_padawans("R2Q4")) > 1
