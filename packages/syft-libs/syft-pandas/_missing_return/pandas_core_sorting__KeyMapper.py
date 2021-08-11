#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.sorting._KeyMapper

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.sorting._KeyMapper()
    return obj


# In[2]:


# pandas.core.sorting._KeyMapper._populate_tables
try:
    obj = class_constructor()
    ret = obj._populate_tables()
    type_pandas_core_sorting__KeyMapper__populate_tables = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.sorting._KeyMapper._populate_tables:",
        type_pandas_core_sorting__KeyMapper__populate_tables)
except Exception as e:
    type_pandas_core_sorting__KeyMapper__populate_tables = '_syft_missing'
    print('❌ pandas.core.sorting._KeyMapper._populate_tables: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.core.sorting._KeyMapper.get_key
try:
    obj = class_constructor()
    ret = obj.get_key()
    type_pandas_core_sorting__KeyMapper_get_key = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.sorting._KeyMapper.get_key:",
        type_pandas_core_sorting__KeyMapper_get_key)
except Exception as e:
    type_pandas_core_sorting__KeyMapper_get_key = '_syft_missing'
    print('❌ pandas.core.sorting._KeyMapper.get_key: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

