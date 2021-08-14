#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.apply.Apply

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.apply.Apply()
    return obj


# In[2]:


# pandas.core.apply.Apply._try_aggregate_string_function
try:
    obj = class_constructor()
    ret = obj._try_aggregate_string_function()
    type_pandas_core_apply_Apply__try_aggregate_string_function = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.Apply._try_aggregate_string_function:",
        type_pandas_core_apply_Apply__try_aggregate_string_function,
    )
except Exception as e:
    type_pandas_core_apply_Apply__try_aggregate_string_function = "_syft_missing"
    print(
        "❌ pandas.core.apply.Apply._try_aggregate_string_function: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.apply.Apply.transform_dict_like
try:
    obj = class_constructor()
    ret = obj.transform_dict_like()
    type_pandas_core_apply_Apply_transform_dict_like = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.apply.Apply.transform_dict_like:",
        type_pandas_core_apply_Apply_transform_dict_like,
    )
except Exception as e:
    type_pandas_core_apply_Apply_transform_dict_like = "_syft_missing"
    print("❌ pandas.core.apply.Apply.transform_dict_like: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
