#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.describe.DataFrameDescriber

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.describe.DataFrameDescriber()
    return obj


# In[2]:


# pandas.core.describe.DataFrameDescriber._select_data
try:
    obj = class_constructor()
    ret = obj._select_data()
    type_pandas_core_describe_DataFrameDescriber__select_data = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.describe.DataFrameDescriber._select_data:",
        type_pandas_core_describe_DataFrameDescriber__select_data,
    )
except Exception as e:
    type_pandas_core_describe_DataFrameDescriber__select_data = "_syft_missing"
    print("❌ pandas.core.describe.DataFrameDescriber._select_data: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
