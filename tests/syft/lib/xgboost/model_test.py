# third party
import pytest

# syft absolute
import syft as sy


@pytest.mark.vendor(lib="xgboost")
def test_xgb_base_module(root_client: sy.VirtualMachineClient) -> None:
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

    classifier = xgb_remote.XGBClassifier(
        n_estimators=100, reg_lambda=1, gamma=0, max_depth=3
    )

    classifier.fit(X, y)
    y_pred_classifier_remote = classifier.predict(X).get(root_client)

    classifier = xgb.XGBClassifier(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)

    classifier.fit(X, y)
    y_pred_classifier = classifier.predict(X)

    regressor = xgb.XGBRegressor(n_estimators=100, reg_lambda=1, gamma=0, max_depth=3)
    regressor.fit(X, y)
    y_pred_regressor = regressor.predict(X)

    regressor = xgb_remote.XGBRegressor(
        n_estimators=100, reg_lambda=1, gamma=0, max_depth=3
    )
    regressor.fit(X, y)
    y_pred_regressor_remote = regressor.predict(X).get(root_client)

    D_train = xgb_remote.DMatrix(X, label=y)
    model = xgb_remote.train(param, D_train, steps)
    preds_remote = model.predict(D_train).get(root_client)

    assert y_pred_regressor.all() == y_pred_regressor_remote.all()
    assert y_pred_classifier_remote.all() == y_pred_classifier.all()
    assert preds_remote.all() == preds.all()
