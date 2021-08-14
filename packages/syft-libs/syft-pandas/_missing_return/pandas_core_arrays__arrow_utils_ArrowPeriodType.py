#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.arrays._arrow_utils.ArrowPeriodType

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.arrays._arrow_utils.ArrowPeriodType()
    return obj


# In[2]:


# pandas.core.arrays._arrow_utils.ArrowPeriodType.__arrow_ext_deserialize__
try:
    obj = class_constructor()
    ret = obj.__arrow_ext_deserialize__()
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___arrow_ext_deserialize__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.arrays._arrow_utils.ArrowPeriodType.__arrow_ext_deserialize__:",
        type_pandas_core_arrays__arrow_utils_ArrowPeriodType___arrow_ext_deserialize__,
    )
except Exception as e:
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___arrow_ext_deserialize__ = (
        "_syft_missing"
    )
    print(
        "❌ pandas.core.arrays._arrow_utils.ArrowPeriodType.__arrow_ext_deserialize__: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.arrays._arrow_utils.ArrowPeriodType.__arrow_ext_serialize__
try:
    obj = class_constructor()
    ret = obj.__arrow_ext_serialize__()
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___arrow_ext_serialize__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.arrays._arrow_utils.ArrowPeriodType.__arrow_ext_serialize__:",
        type_pandas_core_arrays__arrow_utils_ArrowPeriodType___arrow_ext_serialize__,
    )
except Exception as e:
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___arrow_ext_serialize__ = (
        "_syft_missing"
    )
    print(
        "❌ pandas.core.arrays._arrow_utils.ArrowPeriodType.__arrow_ext_serialize__: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas.core.arrays._arrow_utils.ArrowPeriodType.__eq__
try:
    obj = class_constructor()
    ret = obj.__eq__()
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___eq__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.arrays._arrow_utils.ArrowPeriodType.__eq__:",
        type_pandas_core_arrays__arrow_utils_ArrowPeriodType___eq__,
    )
except Exception as e:
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___eq__ = "_syft_missing"
    print(
        "❌ pandas.core.arrays._arrow_utils.ArrowPeriodType.__eq__: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[5]:


# pandas.core.arrays._arrow_utils.ArrowPeriodType.__hash__
try:
    obj = class_constructor()
    ret = obj.__hash__()
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___hash__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.arrays._arrow_utils.ArrowPeriodType.__hash__:",
        type_pandas_core_arrays__arrow_utils_ArrowPeriodType___hash__,
    )
except Exception as e:
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType___hash__ = "_syft_missing"
    print(
        "❌ pandas.core.arrays._arrow_utils.ArrowPeriodType.__hash__: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[6]:


# pandas.core.arrays._arrow_utils.ArrowPeriodType.freq
try:
    obj = class_constructor()
    ret = obj.freq
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType_freq = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.arrays._arrow_utils.ArrowPeriodType.freq:",
        type_pandas_core_arrays__arrow_utils_ArrowPeriodType_freq,
    )
except Exception as e:
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType_freq = "_syft_missing"
    print("❌ pandas.core.arrays._arrow_utils.ArrowPeriodType.freq: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[7]:


# pandas.core.arrays._arrow_utils.ArrowPeriodType.to_pandas_dtype
try:
    obj = class_constructor()
    ret = obj.to_pandas_dtype()
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType_to_pandas_dtype = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.arrays._arrow_utils.ArrowPeriodType.to_pandas_dtype:",
        type_pandas_core_arrays__arrow_utils_ArrowPeriodType_to_pandas_dtype,
    )
except Exception as e:
    type_pandas_core_arrays__arrow_utils_ArrowPeriodType_to_pandas_dtype = (
        "_syft_missing"
    )
    print(
        "❌ pandas.core.arrays._arrow_utils.ArrowPeriodType.to_pandas_dtype: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
