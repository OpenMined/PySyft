#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.groupby.grouper.Grouping

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.groupby.grouper.Grouping()
    return obj


# In[2]:


# pandas.core.groupby.grouper.Grouping.__iter__
try:
    obj = class_constructor()
    ret = obj.__iter__()
    type_pandas_core_groupby_grouper_Grouping___iter__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.groupby.grouper.Grouping.__iter__:",
        type_pandas_core_groupby_grouper_Grouping___iter__)
except Exception as e:
    type_pandas_core_groupby_grouper_Grouping___iter__ = '_syft_missing'
    print('❌ pandas.core.groupby.grouper.Grouping.__iter__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.core.groupby.grouper.Grouping.codes
try:
    obj = class_constructor()
    ret = obj.codes
    type_pandas_core_groupby_grouper_Grouping_codes = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.groupby.grouper.Grouping.codes:",
        type_pandas_core_groupby_grouper_Grouping_codes)
except Exception as e:
    type_pandas_core_groupby_grouper_Grouping_codes = '_syft_missing'
    print('❌ pandas.core.groupby.grouper.Grouping.codes: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[4]:


# pandas.core.groupby.grouper.Grouping.ngroups
try:
    obj = class_constructor()
    ret = obj.ngroups
    type_pandas_core_groupby_grouper_Grouping_ngroups = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.groupby.grouper.Grouping.ngroups:",
        type_pandas_core_groupby_grouper_Grouping_ngroups)
except Exception as e:
    type_pandas_core_groupby_grouper_Grouping_ngroups = '_syft_missing'
    print('❌ pandas.core.groupby.grouper.Grouping.ngroups: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

