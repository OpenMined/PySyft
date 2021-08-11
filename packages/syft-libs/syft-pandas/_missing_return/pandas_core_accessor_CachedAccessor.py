#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.accessor.CachedAccessor

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.accessor.CachedAccessor()
    return obj


# In[2]:


# pandas.core.accessor.CachedAccessor.__get__
try:
    obj = class_constructor()
    ret = obj.__get__()
    type_pandas_core_accessor_CachedAccessor___get__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.accessor.CachedAccessor.__get__:",
        type_pandas_core_accessor_CachedAccessor___get__)
except Exception as e:
    type_pandas_core_accessor_CachedAccessor___get__ = '_syft_missing'
    print('❌ pandas.core.accessor.CachedAccessor.__get__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[ ]:




