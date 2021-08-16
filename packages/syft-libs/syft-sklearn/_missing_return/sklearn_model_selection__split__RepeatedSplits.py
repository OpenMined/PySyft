#!/usr/bin/env python
# coding: utf-8

# ## sklearn.model_selection._split._RepeatedSplits

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.model_selection._split._RepeatedSplits()
    return obj


# In[ ]:


# sklearn.model_selection._split._RepeatedSplits.__repr__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__repr__()
    type_sklearn_model_selection__split__RepeatedSplits___repr__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.model_selection._split._RepeatedSplits.__repr__: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_model_selection__split__RepeatedSplits___repr__ = "_syft_missing"
    print(
        "❌ sklearn.model_selection._split._RepeatedSplits.__repr__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.model_selection._split._RepeatedSplits.get_n_splits
try:
    obj = class_constructor()  # noqa F821
    ret = obj.get_n_splits()
    type_sklearn_model_selection__split__RepeatedSplits_get_n_splits = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.model_selection._split._RepeatedSplits.get_n_splits: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_model_selection__split__RepeatedSplits_get_n_splits = "_syft_missing"
    print(
        "❌ sklearn.model_selection._split._RepeatedSplits.get_n_splits: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.model_selection._split._RepeatedSplits.split
try:
    obj = class_constructor()  # noqa F821
    ret = obj.split()
    type_sklearn_model_selection__split__RepeatedSplits_split = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.model_selection._split._RepeatedSplits.split: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_model_selection__split__RepeatedSplits_split = "_syft_missing"
    print(
        "❌ sklearn.model_selection._split._RepeatedSplits.split: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
