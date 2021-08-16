#!/usr/bin/env python
# coding: utf-8

# ## sklearn.utils._encode._nandict

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.utils._encode._nandict()
    return obj


# In[ ]:


# sklearn.utils._encode._nandict.__missing__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__missing__()
    type_sklearn_utils__encode__nandict___missing__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.utils._encode._nandict.__missing__: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_utils__encode__nandict___missing__ = "_syft_missing"
    print(
        "❌ sklearn.utils._encode._nandict.__missing__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
