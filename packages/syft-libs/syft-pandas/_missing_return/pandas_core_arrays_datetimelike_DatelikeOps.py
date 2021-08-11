#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.arrays.datetimelike.DatelikeOps

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.arrays.datetimelike.DatelikeOps()
    return obj


# In[2]:


# pandas.core.arrays.datetimelike.DatelikeOps.strftime
try:
    obj = class_constructor()
    ret = obj.strftime('%%B %%d, %%Y, %%r')
    type_pandas_core_arrays_datetimelike_DatelikeOps_strftime = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.datetimelike.DatelikeOps.strftime:",
        type_pandas_core_arrays_datetimelike_DatelikeOps_strftime)
except Exception as e:
    type_pandas_core_arrays_datetimelike_DatelikeOps_strftime = '_syft_missing'
    print('❌ pandas.core.arrays.datetimelike.DatelikeOps.strftime: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[ ]:




