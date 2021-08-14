#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.reshape.merge._OrderedMerge

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.reshape.merge._OrderedMerge()
    return obj


# In[2]:


# pandas.core.reshape.merge._OrderedMerge._get_merge_keys
try:
    obj = class_constructor()
    ret = obj._get_merge_keys()
    type_pandas_core_reshape_merge__OrderedMerge__get_merge_keys = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.reshape.merge._OrderedMerge._get_merge_keys:",
        type_pandas_core_reshape_merge__OrderedMerge__get_merge_keys)
except Exception as e:
    type_pandas_core_reshape_merge__OrderedMerge__get_merge_keys = '_syft_missing'
    print('❌ pandas.core.reshape.merge._OrderedMerge._get_merge_keys: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

