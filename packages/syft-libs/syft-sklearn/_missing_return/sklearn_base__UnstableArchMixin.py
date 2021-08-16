#!/usr/bin/env python
# coding: utf-8

# ## sklearn.base._UnstableArchMixin

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.base._UnstableArchMixin()
    return obj


# In[ ]:


# sklearn.base._UnstableArchMixin._more_tags
try:
    obj = class_constructor()  # noqa F821
    ret = obj._more_tags()
    type_sklearn_base__UnstableArchMixin__more_tags = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.base._UnstableArchMixin._more_tags: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_base__UnstableArchMixin__more_tags = "_syft_missing"
    print(
        "❌ sklearn.base._UnstableArchMixin._more_tags: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
