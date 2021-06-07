# third party
import pytest

# syft absolute
import syft as sy
from syft.experimental_flags import flags

np = pytest.importorskip("numpy")
sy.load("numpy")
sklearn = pytest.importorskip("sklearn")
sy.load("sklearn")


@pytest.mark.vendor(lib="sklearn")
@pytest.mark.parametrize("arrow_backend", [True, False])
def test_logistic_model_serde(
    root_client: sy.VirtualMachineClient, arrow_backend: bool
) -> None:
    # third party
    from sklearn.linear_model import LogisticRegression

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
