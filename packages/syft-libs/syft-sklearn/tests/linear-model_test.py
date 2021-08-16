# third party
import pytest

# syft absolute
import syft as sy
from syft.experimental_flags import flags

import syft_sklearn  # noqa F401 # isort:skip

np = pytest.importorskip("numpy")
linear_model = pytest.importorskip("sklearn.linear_model")
LogisticRegression = linear_model.LogisticRegression


@pytest.mark.vendor(lib="sklearn")
@pytest.mark.parametrize("arrow_backend", [True, False])
def test_logistic(arrow_backend: bool, root_client: sy.VirtualMachineClient) -> None:
    flags.APACHE_ARROW_TENSOR_SERDE = arrow_backend
    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([0, 0, 1, 1])
    clf = LogisticRegression(random_state=0).fit(X, y)

    LR_remote = root_client.sklearn.linear_model.LogisticRegression

    clf2 = LR_remote(random_state=0).fit(X, y)

    assert clf.predict(X).all() == clf2.predict(X).get().all()
