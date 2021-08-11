#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.internals.concat.JoinUnit

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.internals.concat.JoinUnit()
    return obj


# In[2]:


# pandas.core.internals.concat.JoinUnit.get_reindexed_values
try:
    obj = class_constructor()
    ret = obj.get_reindexed_values()
    type_pandas_core_internals_concat_JoinUnit_get_reindexed_values = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.internals.concat.JoinUnit.get_reindexed_values:",
        type_pandas_core_internals_concat_JoinUnit_get_reindexed_values)
except Exception as e:
    type_pandas_core_internals_concat_JoinUnit_get_reindexed_values = '_syft_missing'
    print('❌ pandas.core.internals.concat.JoinUnit.get_reindexed_values: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

