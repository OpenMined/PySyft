#Mikaela
# "Do or do not. There is no try." — Yoda

def get_padawans(cohort):
    # add yourself to the temple trial roster
    data = {
        "R2Q4": {"skywalker": "PASSED" }}
    return data[cohort]


def test_trial_of_skill():
    assert get_padawans("R2Q4")["skywalker"]  == "PASSED"
    assert len(get_padawans("R2Q4")) == 1
