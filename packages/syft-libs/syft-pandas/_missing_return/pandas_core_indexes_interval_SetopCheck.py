#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.indexes.interval.SetopCheck

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.indexes.interval.SetopCheck()
    return obj


# In[2]:


# pandas.core.indexes.interval.SetopCheck.__call__
try:
    obj = class_constructor()
    ret = obj.__call__()
    type_pandas_core_indexes_interval_SetopCheck___call__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.indexes.interval.SetopCheck.__call__:",
        type_pandas_core_indexes_interval_SetopCheck___call__)
except Exception as e:
    type_pandas_core_indexes_interval_SetopCheck___call__ = '_syft_missing'
    print('❌ pandas.core.indexes.interval.SetopCheck.__call__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

