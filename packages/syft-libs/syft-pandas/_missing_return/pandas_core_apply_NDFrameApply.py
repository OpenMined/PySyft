#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.apply.NDFrameApply

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.apply.NDFrameApply()
    return obj


# In[2]:


# pandas.core.apply.NDFrameApply._try_aggregate_string_function
try:
    obj = class_constructor()
    ret = obj._try_aggregate_string_function()
    type_pandas_core_apply_NDFrameApply__try_aggregate_string_function = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.NDFrameApply._try_aggregate_string_function:",
        type_pandas_core_apply_NDFrameApply__try_aggregate_string_function,
    )
except Exception as e:
    type_pandas_core_apply_NDFrameApply__try_aggregate_string_function = "_syft_missing"
    print(
        "❌ pandas.core.apply.NDFrameApply._try_aggregate_string_function: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.apply.NDFrameApply.agg_axis
try:
    obj = class_constructor()
    ret = obj.agg_axis
    type_pandas_core_apply_NDFrameApply_agg_axis = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.NDFrameApply.agg_axis:",
        type_pandas_core_apply_NDFrameApply_agg_axis,
    )
except Exception as e:
    type_pandas_core_apply_NDFrameApply_agg_axis = "_syft_missing"
    print("❌ pandas.core.apply.NDFrameApply.agg_axis: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[4]:


# pandas.core.apply.NDFrameApply.index
try:
    obj = class_constructor()
    ret = obj.index
    type_pandas_core_apply_NDFrameApply_index = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.NDFrameApply.index:",
        type_pandas_core_apply_NDFrameApply_index,
    )
except Exception as e:
    type_pandas_core_apply_NDFrameApply_index = "_syft_missing"
    print("❌ pandas.core.apply.NDFrameApply.index: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[5]:


# pandas.core.apply.NDFrameApply.transform_dict_like
try:
    obj = class_constructor()
    ret = obj.transform_dict_like()
    type_pandas_core_apply_NDFrameApply_transform_dict_like = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.NDFrameApply.transform_dict_like:",
        type_pandas_core_apply_NDFrameApply_transform_dict_like,
    )
except Exception as e:
    type_pandas_core_apply_NDFrameApply_transform_dict_like = "_syft_missing"
    print("❌ pandas.core.apply.NDFrameApply.transform_dict_like: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
