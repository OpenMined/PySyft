#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.apply.GroupByApply

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.apply.GroupByApply()
    return obj


# In[2]:


# pandas.core.apply.GroupByApply._try_aggregate_string_function
try:
    obj = class_constructor()
    ret = obj._try_aggregate_string_function()
    type_pandas_core_apply_GroupByApply__try_aggregate_string_function = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.GroupByApply._try_aggregate_string_function:",
        type_pandas_core_apply_GroupByApply__try_aggregate_string_function,
    )
except Exception as e:
    type_pandas_core_apply_GroupByApply__try_aggregate_string_function = "_syft_missing"
    print(
        "❌ pandas.core.apply.GroupByApply._try_aggregate_string_function: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.apply.GroupByApply.apply
try:
    obj = class_constructor()
    ret = obj.apply()
    type_pandas_core_apply_GroupByApply_apply = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.GroupByApply.apply:",
        type_pandas_core_apply_GroupByApply_apply,
    )
except Exception as e:
    type_pandas_core_apply_GroupByApply_apply = "_syft_missing"
    print("❌ pandas.core.apply.GroupByApply.apply: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas.core.apply.GroupByApply.transform
try:
    obj = class_constructor()
    ret = obj.transform()
    type_pandas_core_apply_GroupByApply_transform = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.GroupByApply.transform:",
        type_pandas_core_apply_GroupByApply_transform,
    )
except Exception as e:
    type_pandas_core_apply_GroupByApply_transform = "_syft_missing"
    print("❌ pandas.core.apply.GroupByApply.transform: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[5]:


# pandas.core.apply.GroupByApply.transform_dict_like
try:
    obj = class_constructor()
    ret = obj.transform_dict_like()
    type_pandas_core_apply_GroupByApply_transform_dict_like = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.GroupByApply.transform_dict_like:",
        type_pandas_core_apply_GroupByApply_transform_dict_like,
    )
except Exception as e:
    type_pandas_core_apply_GroupByApply_transform_dict_like = "_syft_missing"
    print("❌ pandas.core.apply.GroupByApply.transform_dict_like: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
