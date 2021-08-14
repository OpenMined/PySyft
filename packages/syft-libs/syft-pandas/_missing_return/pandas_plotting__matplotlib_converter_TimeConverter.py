#!/usr/bin/env python
# coding: utf-8

# ## pandas.plotting._matplotlib.converter.TimeConverter

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.plotting._matplotlib.converter.TimeConverter()
    return obj


# In[2]:


# pandas.plotting._matplotlib.converter.TimeConverter.convert
try:
    obj = class_constructor()
    ret = obj.convert()
    type_pandas_plotting__matplotlib_converter_TimeConverter_convert = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.plotting._matplotlib.converter.TimeConverter.convert:",
        type_pandas_plotting__matplotlib_converter_TimeConverter_convert,
    )
except Exception as e:
    type_pandas_plotting__matplotlib_converter_TimeConverter_convert = "_syft_missing"
    print(
        "❌ pandas.plotting._matplotlib.converter.TimeConverter.convert: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.plotting._matplotlib.converter.TimeConverter.is_numlike
try:
    obj = class_constructor()
    ret = obj.is_numlike()
    type_pandas_plotting__matplotlib_converter_TimeConverter_is_numlike = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.plotting._matplotlib.converter.TimeConverter.is_numlike:",
        type_pandas_plotting__matplotlib_converter_TimeConverter_is_numlike,
    )
except Exception as e:
    type_pandas_plotting__matplotlib_converter_TimeConverter_is_numlike = (
        "_syft_missing"
    )
    print(
        "❌ pandas.plotting._matplotlib.converter.TimeConverter.is_numlike: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
