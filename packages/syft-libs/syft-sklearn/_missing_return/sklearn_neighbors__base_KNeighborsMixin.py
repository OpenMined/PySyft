#!/usr/bin/env python
# coding: utf-8

# ## sklearn.neighbors._base.KNeighborsMixin

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.neighbors._base.KNeighborsMixin()
    return obj


# In[ ]:


# sklearn.neighbors._base.KNeighborsMixin._kneighbors_reduce_func
try:
    obj = class_constructor()  # noqa F821
    ret = obj._kneighbors_reduce_func()
    type_sklearn_neighbors__base_KNeighborsMixin__kneighbors_reduce_func = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.neighbors._base.KNeighborsMixin._kneighbors_reduce_func: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_neighbors__base_KNeighborsMixin__kneighbors_reduce_func = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.neighbors._base.KNeighborsMixin._kneighbors_reduce_func: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.neighbors._base.KNeighborsMixin.kneighbors
try:
    obj = class_constructor()  # noqa F821
    ret = obj.kneighbors()
    type_sklearn_neighbors__base_KNeighborsMixin_kneighbors = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.neighbors._base.KNeighborsMixin.kneighbors: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_neighbors__base_KNeighborsMixin_kneighbors = "_syft_missing"
    print(
        "❌ sklearn.neighbors._base.KNeighborsMixin.kneighbors: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.neighbors._base.KNeighborsMixin.kneighbors_graph
try:
    obj = class_constructor()  # noqa F821
    ret = obj.kneighbors_graph()
    type_sklearn_neighbors__base_KNeighborsMixin_kneighbors_graph = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.neighbors._base.KNeighborsMixin.kneighbors_graph: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_neighbors__base_KNeighborsMixin_kneighbors_graph = "_syft_missing"
    print(
        "❌ sklearn.neighbors._base.KNeighborsMixin.kneighbors_graph: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
