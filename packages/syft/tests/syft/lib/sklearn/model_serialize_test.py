# third party
import pytest

# syft absolute
import syft as sy
from syft.experimental_flags import flags


@pytest.mark.vendor(lib="sklearn")
@pytest.mark.parametrize("arrow_backend", [True, False])
def test_logistic_model_serde(arrow_backend: bool) -> None:
    # Don't share with other tests due to the _regenerate_numpy_serde that occurs with
    # flags.APACHE_ARROW_TENSOR_SERDE = arrow_backend
    vm = sy.VirtualMachine()
    root_client = vm.get_root_client()

    # third party
    import numpy as np
    from sklearn.linear_model import LogisticRegression

    sy.load("sklearn")
    sy.load("numpy")

    flags.APACHE_ARROW_TENSOR_SERDE = arrow_backend
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([0, 0, 1, 1])
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
