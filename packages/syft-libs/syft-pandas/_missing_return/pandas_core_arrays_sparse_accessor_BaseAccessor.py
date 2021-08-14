#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.arrays.sparse.accessor.BaseAccessor

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.arrays.sparse.accessor.BaseAccessor()
    return obj


# In[2]:


# pandas.core.arrays.sparse.accessor.BaseAccessor._validate
try:
    obj = class_constructor()
    ret = obj._validate()
    type_pandas_core_arrays_sparse_accessor_BaseAccessor__validate = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.arrays.sparse.accessor.BaseAccessor._validate:",
        type_pandas_core_arrays_sparse_accessor_BaseAccessor__validate,
    )
except Exception as e:
    type_pandas_core_arrays_sparse_accessor_BaseAccessor__validate = "_syft_missing"
    print(
        "❌ pandas.core.arrays.sparse.accessor.BaseAccessor._validate: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
