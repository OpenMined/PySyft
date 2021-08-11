#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.indexing._IndexSlice

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.indexing._IndexSlice()
    return obj


# In[2]:


# pandas.core.indexing._IndexSlice.__getitem__
try:
    obj = class_constructor()
    ret = obj.__getitem__()
    type_pandas_core_indexing__IndexSlice___getitem__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.indexing._IndexSlice.__getitem__:",
        type_pandas_core_indexing__IndexSlice___getitem__)
except Exception as e:
    type_pandas_core_indexing__IndexSlice___getitem__ = '_syft_missing'
    print('❌ pandas.core.indexing._IndexSlice.__getitem__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

