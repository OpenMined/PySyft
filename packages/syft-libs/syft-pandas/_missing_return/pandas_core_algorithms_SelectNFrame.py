#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.algorithms.SelectNFrame

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.algorithms.SelectNFrame()
    return obj


# In[2]:


# pandas.core.algorithms.SelectNFrame.nlargest
try:
    obj = class_constructor()
    ret = obj.nlargest()
    type_pandas_core_algorithms_SelectNFrame_nlargest = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.algorithms.SelectNFrame.nlargest:",
        type_pandas_core_algorithms_SelectNFrame_nlargest,
    )
except Exception as e:
    type_pandas_core_algorithms_SelectNFrame_nlargest = "_syft_missing"
    print("❌ pandas.core.algorithms.SelectNFrame.nlargest: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.algorithms.SelectNFrame.nsmallest
try:
    obj = class_constructor()
    ret = obj.nsmallest()
    type_pandas_core_algorithms_SelectNFrame_nsmallest = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.algorithms.SelectNFrame.nsmallest:",
        type_pandas_core_algorithms_SelectNFrame_nsmallest,
    )
except Exception as e:
    type_pandas_core_algorithms_SelectNFrame_nsmallest = "_syft_missing"
    print("❌ pandas.core.algorithms.SelectNFrame.nsmallest: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
