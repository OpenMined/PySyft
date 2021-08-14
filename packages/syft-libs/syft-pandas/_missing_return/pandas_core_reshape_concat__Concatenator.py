#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.reshape.concat._Concatenator

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.reshape.concat._Concatenator()
    return obj


# In[2]:


# pandas.core.reshape.concat._Concatenator._maybe_check_integrity
try:
    obj = class_constructor()
    ret = obj._maybe_check_integrity()
    type_pandas_core_reshape_concat__Concatenator__maybe_check_integrity = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.reshape.concat._Concatenator._maybe_check_integrity:",
        type_pandas_core_reshape_concat__Concatenator__maybe_check_integrity,
    )
except Exception as e:
    type_pandas_core_reshape_concat__Concatenator__maybe_check_integrity = (
        "_syft_missing"
    )
    print(
        "❌ pandas.core.reshape.concat._Concatenator._maybe_check_integrity: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.reshape.concat._Concatenator.get_result
try:
    obj = class_constructor()
    ret = obj.get_result()
    type_pandas_core_reshape_concat__Concatenator_get_result = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.reshape.concat._Concatenator.get_result:",
        type_pandas_core_reshape_concat__Concatenator_get_result,
    )
except Exception as e:
    type_pandas_core_reshape_concat__Concatenator_get_result = "_syft_missing"
    print("❌ pandas.core.reshape.concat._Concatenator.get_result: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
