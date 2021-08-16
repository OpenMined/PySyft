#!/usr/bin/env python
# coding: utf-8

# ## sklearn.utils.fixes._FuncWrapper

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.utils.fixes._FuncWrapper()
    return obj


# In[ ]:


# sklearn.utils.fixes._FuncWrapper.__call__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__call__()
    type_sklearn_utils_fixes__FuncWrapper___call__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.utils.fixes._FuncWrapper.__call__: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_utils_fixes__FuncWrapper___call__ = "_syft_missing"
    print(
        "❌ sklearn.utils.fixes._FuncWrapper.__call__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
