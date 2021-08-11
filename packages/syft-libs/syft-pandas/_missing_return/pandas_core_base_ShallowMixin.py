#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.base.ShallowMixin

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.base.ShallowMixin()
    return obj


# In[2]:


# pandas.core.base.ShallowMixin._shallow_copy
try:
    obj = class_constructor()
    ret = obj._shallow_copy()
    type_pandas_core_base_ShallowMixin__shallow_copy = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.base.ShallowMixin._shallow_copy:",
        type_pandas_core_base_ShallowMixin__shallow_copy)
except Exception as e:
    type_pandas_core_base_ShallowMixin__shallow_copy = '_syft_missing'
    print('❌ pandas.core.base.ShallowMixin._shallow_copy: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

