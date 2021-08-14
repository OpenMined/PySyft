#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.indexing._AtIndexer

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.indexing._AtIndexer()
    return obj


# In[2]:


# pandas.core.indexing._AtIndexer.__getitem__
try:
    obj = class_constructor()
    ret = obj.__getitem__()
    type_pandas_core_indexing__AtIndexer___getitem__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.indexing._AtIndexer.__getitem__:",
        type_pandas_core_indexing__AtIndexer___getitem__,
    )
except Exception as e:
    type_pandas_core_indexing__AtIndexer___getitem__ = "_syft_missing"
    print("❌ pandas.core.indexing._AtIndexer.__getitem__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.indexing._AtIndexer.__setitem__
try:
    obj = class_constructor()
    ret = obj.__setitem__()
    type_pandas_core_indexing__AtIndexer___setitem__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.indexing._AtIndexer.__setitem__:",
        type_pandas_core_indexing__AtIndexer___setitem__,
    )
except Exception as e:
    type_pandas_core_indexing__AtIndexer___setitem__ = "_syft_missing"
    print("❌ pandas.core.indexing._AtIndexer.__setitem__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas.core.indexing._AtIndexer._axes_are_unique
try:
    obj = class_constructor()
    ret = obj._axes_are_unique
    type_pandas_core_indexing__AtIndexer__axes_are_unique = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.indexing._AtIndexer._axes_are_unique:",
        type_pandas_core_indexing__AtIndexer__axes_are_unique,
    )
except Exception as e:
    type_pandas_core_indexing__AtIndexer__axes_are_unique = "_syft_missing"
    print("❌ pandas.core.indexing._AtIndexer._axes_are_unique: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[5]:


# pandas.core.indexing._AtIndexer._convert_key
try:
    obj = class_constructor()
    ret = obj._convert_key()
    type_pandas_core_indexing__AtIndexer__convert_key = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.indexing._AtIndexer._convert_key:",
        type_pandas_core_indexing__AtIndexer__convert_key,
    )
except Exception as e:
    type_pandas_core_indexing__AtIndexer__convert_key = "_syft_missing"
    print("❌ pandas.core.indexing._AtIndexer._convert_key: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
