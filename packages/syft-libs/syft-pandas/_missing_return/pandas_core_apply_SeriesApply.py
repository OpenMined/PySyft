#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.apply.SeriesApply

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.apply.SeriesApply()
    return obj


# In[2]:


# pandas.core.apply.SeriesApply._try_aggregate_string_function
try:
    obj = class_constructor()
    ret = obj._try_aggregate_string_function()
    type_pandas_core_apply_SeriesApply__try_aggregate_string_function = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.SeriesApply._try_aggregate_string_function:",
        type_pandas_core_apply_SeriesApply__try_aggregate_string_function,
    )
except Exception as e:
    type_pandas_core_apply_SeriesApply__try_aggregate_string_function = "_syft_missing"
    print(
        "❌ pandas.core.apply.SeriesApply._try_aggregate_string_function: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.apply.SeriesApply.agg
try:
    obj = class_constructor()
    ret = obj.agg()
    type_pandas_core_apply_SeriesApply_agg = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.SeriesApply.agg:", type_pandas_core_apply_SeriesApply_agg
    )
except Exception as e:
    type_pandas_core_apply_SeriesApply_agg = "_syft_missing"
    print("❌ pandas.core.apply.SeriesApply.agg: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas.core.apply.SeriesApply.agg_axis
try:
    obj = class_constructor()
    ret = obj.agg_axis
    type_pandas_core_apply_SeriesApply_agg_axis = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.SeriesApply.agg_axis:",
        type_pandas_core_apply_SeriesApply_agg_axis,
    )
except Exception as e:
    type_pandas_core_apply_SeriesApply_agg_axis = "_syft_missing"
    print("❌ pandas.core.apply.SeriesApply.agg_axis: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[5]:


# pandas.core.apply.SeriesApply.index
try:
    obj = class_constructor()
    ret = obj.index
    type_pandas_core_apply_SeriesApply_index = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.SeriesApply.index:",
        type_pandas_core_apply_SeriesApply_index,
    )
except Exception as e:
    type_pandas_core_apply_SeriesApply_index = "_syft_missing"
    print("❌ pandas.core.apply.SeriesApply.index: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[6]:


# pandas.core.apply.SeriesApply.transform_dict_like
try:
    obj = class_constructor()
    ret = obj.transform_dict_like()
    type_pandas_core_apply_SeriesApply_transform_dict_like = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.SeriesApply.transform_dict_like:",
        type_pandas_core_apply_SeriesApply_transform_dict_like,
    )
except Exception as e:
    type_pandas_core_apply_SeriesApply_transform_dict_like = "_syft_missing"
    print("❌ pandas.core.apply.SeriesApply.transform_dict_like: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
