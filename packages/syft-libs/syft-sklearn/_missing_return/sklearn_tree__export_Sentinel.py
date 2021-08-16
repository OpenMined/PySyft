#!/usr/bin/env python
# coding: utf-8

# ## sklearn.tree._export.Sentinel

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.tree._export.Sentinel()
    return obj


# In[ ]:


# sklearn.tree._export.Sentinel.__repr__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__repr__()
    type_sklearn_tree__export_Sentinel___repr__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.tree._export.Sentinel.__repr__: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_tree__export_Sentinel___repr__ = "_syft_missing"
    print("❌ sklearn.tree._export.Sentinel.__repr__: Return unavailable")  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
