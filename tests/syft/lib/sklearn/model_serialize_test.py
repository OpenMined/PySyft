# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="sklearn")
def test_logistic_model_serde(root_client: sy.VirtualMachineClient) -> None:

    sy.load("sklearn")
    sy.load("numpy")

    # third party
    import numpy as np

    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([0, 0, 1, 1])

    # third party
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(random_state=0).fit(X, y)

    clf_remote = clf.send(root_client)

    clf_2 = clf_remote.get()

    dict_1 = vars(clf)
    dict_2 = vars(clf_2)

    for key in dict_1.keys():
        if type(dict_1[key]) == float:
            assert abs(dict_1[key] - dict_2[key]) < 0.0001
        elif type(dict_1[key]) == np.ndarray:
            assert dict_1[key].all() == dict_2[key].all()
        else:
            assert dict_1[key] == dict_2[key]
