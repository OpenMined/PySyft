#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.groupby.ops.SeriesSplitter

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.groupby.ops.SeriesSplitter()
    return obj


# In[2]:


# pandas.core.groupby.ops.SeriesSplitter.__iter__
try:
    obj = class_constructor()
    ret = obj.__iter__()
    type_pandas_core_groupby_ops_SeriesSplitter___iter__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.groupby.ops.SeriesSplitter.__iter__:",
        type_pandas_core_groupby_ops_SeriesSplitter___iter__)
except Exception as e:
    type_pandas_core_groupby_ops_SeriesSplitter___iter__ = '_syft_missing'
    print('❌ pandas.core.groupby.ops.SeriesSplitter.__iter__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

