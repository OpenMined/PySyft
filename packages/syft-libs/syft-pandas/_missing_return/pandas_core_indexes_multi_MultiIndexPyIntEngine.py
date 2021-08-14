#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.indexes.multi.MultiIndexPyIntEngine

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.indexes.multi.MultiIndexPyIntEngine()
    return obj


# In[2]:


# pandas.core.indexes.multi.MultiIndexPyIntEngine._codes_to_ints
try:
    obj = class_constructor()
    ret = obj._codes_to_ints()
    type_pandas_core_indexes_multi_MultiIndexPyIntEngine__codes_to_ints = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.indexes.multi.MultiIndexPyIntEngine._codes_to_ints:",
        type_pandas_core_indexes_multi_MultiIndexPyIntEngine__codes_to_ints)
except Exception as e:
    type_pandas_core_indexes_multi_MultiIndexPyIntEngine__codes_to_ints = '_syft_missing'
    print('❌ pandas.core.indexes.multi.MultiIndexPyIntEngine._codes_to_ints: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

