# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="xgboost")
def test_pandas(root_client: sy.VirtualMachineClient) -> None:
    sy.load("xgboost")
    sy.load("numpy")
    # import xgboost

    # third party
    import numpy as np
    import xgboost as xgb

    xgb_remote = root_client.xgboost

    # import xgboost as xgb

    X = np.array([[-1, -1], [-2, -1], [1, 1], [2, 1]])
    y = np.array([1, 1, 2, 2])

    param = {"eta": 0.3, "max_depth": 3, "num_class": 3}

    steps = 20

    D_train = xgb.DMatrix(X, label=y)
    model = xgb.train(param, D_train, steps)
    preds = model.predict(D_train)

    D_train = xgb_remote.DMatrix(X, label=y)
    model = xgb_remote.train(param, D_train, steps)
    preds_remote = model.predict(D_train)

    assert preds_remote.get(root_client).all() == preds.all()
