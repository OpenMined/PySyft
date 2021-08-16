#!/usr/bin/env python
# coding: utf-8

# ## sklearn.calibration._CalibratedClassifier

# In[ ]:


# sklearn.calibration._CalibratedClassifier.calibrators_
try:
    obj = class_constructor()
    ret = obj.calibrators_
    type_sklearn_calibration__CalibratedClassifier_calibrators_ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.calibration._CalibratedClassifier.calibrators_:", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_calibration__CalibratedClassifier_calibrators_ = "_syft_missing"
    print(
        "❌ sklearn.calibration._CalibratedClassifier.calibrators_: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.calibration._CalibratedClassifier.predict_proba
try:
    obj = class_constructor()  # noqa F821
    ret = obj.predict_proba()
    type_sklearn_calibration__CalibratedClassifier_predict_proba = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.calibration._CalibratedClassifier.predict_proba: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_calibration__CalibratedClassifier_predict_proba = "_syft_missing"
    print(
        "❌ sklearn.calibration._CalibratedClassifier.predict_proba: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
