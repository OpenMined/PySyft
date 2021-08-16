#!/usr/bin/env python
# coding: utf-8

# ## sklearn.externals._arff.BadRelationFormat

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.externals._arff.BadRelationFormat()
    return obj


# In[ ]:


# sklearn.externals._arff.BadRelationFormat.__str__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__str__()
    type_sklearn_externals__arff_BadRelationFormat___str__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.externals._arff.BadRelationFormat.__str__: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_externals__arff_BadRelationFormat___str__ = "_syft_missing"
    print(
        "❌ sklearn.externals._arff.BadRelationFormat.__str__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
