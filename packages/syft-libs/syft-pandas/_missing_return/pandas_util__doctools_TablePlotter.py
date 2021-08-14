#!/usr/bin/env python
# coding: utf-8

# ## pandas.util._doctools.TablePlotter

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.util._doctools.TablePlotter()
    return obj


# In[2]:


# pandas.util._doctools.TablePlotter._conv
try:
    obj = class_constructor()
    ret = obj._conv()
    type_pandas_util__doctools_TablePlotter__conv = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.util._doctools.TablePlotter._conv:",
        type_pandas_util__doctools_TablePlotter__conv,
    )
except Exception as e:
    type_pandas_util__doctools_TablePlotter__conv = "_syft_missing"
    print("❌ pandas.util._doctools.TablePlotter._conv: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.util._doctools.TablePlotter._insert_index
try:
    obj = class_constructor()
    ret = obj._insert_index()
    type_pandas_util__doctools_TablePlotter__insert_index = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.util._doctools.TablePlotter._insert_index:",
        type_pandas_util__doctools_TablePlotter__insert_index,
    )
except Exception as e:
    type_pandas_util__doctools_TablePlotter__insert_index = "_syft_missing"
    print("❌ pandas.util._doctools.TablePlotter._insert_index: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas.util._doctools.TablePlotter.plot
try:
    obj = class_constructor()
    ret = obj.plot()
    type_pandas_util__doctools_TablePlotter_plot = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.util._doctools.TablePlotter.plot:",
        type_pandas_util__doctools_TablePlotter_plot,
    )
except Exception as e:
    type_pandas_util__doctools_TablePlotter_plot = "_syft_missing"
    print("❌ pandas.util._doctools.TablePlotter.plot: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
