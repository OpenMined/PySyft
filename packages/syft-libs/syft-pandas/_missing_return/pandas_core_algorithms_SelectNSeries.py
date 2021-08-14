#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.algorithms.SelectNSeries

# In[1]:


# third party
import pandas
import pandas as pd

countries_population = {
    "Italy": 59000000,
    "France": 65000000,
    "Malta": 434000,
    "Maldives": 434000,
    "Brunei": 434000,
    "Iceland": 337000,
    "Nauru": 11300,
    "Tuvalu": 11300,
    "Anguilla": 11300,
    "Montserrat": 5200,
}


s = pd.Series(countries_population)


def class_constructor(*args, **kwargs):
    obj = pandas.core.algorithms.SelectNSeries(s, 2, keep="first")
    return obj


# In[2]:


# pandas.core.algorithms.SelectNSeries.compute
try:
    obj = class_constructor()
    ret = obj.compute("nlargest")
    type_pandas_core_algorithms_SelectNSeries_compute = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.algorithms.SelectNSeries.compute:",
        type_pandas_core_algorithms_SelectNSeries_compute,
    )
except Exception as e:
    type_pandas_core_algorithms_SelectNSeries_compute = "_syft_missing"
    print("❌ pandas.core.algorithms.SelectNSeries.compute: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.algorithms.SelectNSeries.nlargest
try:
    obj = class_constructor()
    ret = obj.nlargest()
    type_pandas_core_algorithms_SelectNSeries_nlargest = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.algorithms.SelectNSeries.nlargest:",
        type_pandas_core_algorithms_SelectNSeries_nlargest,
    )
except Exception as e:
    type_pandas_core_algorithms_SelectNSeries_nlargest = "_syft_missing"
    print("❌ pandas.core.algorithms.SelectNSeries.nlargest: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas.core.algorithms.SelectNSeries.nsmallest
try:
    obj = class_constructor()
    ret = obj.nsmallest()
    type_pandas_core_algorithms_SelectNSeries_nsmallest = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.algorithms.SelectNSeries.nsmallest:",
        type_pandas_core_algorithms_SelectNSeries_nsmallest,
    )
except Exception as e:
    type_pandas_core_algorithms_SelectNSeries_nsmallest = "_syft_missing"
    print("❌ pandas.core.algorithms.SelectNSeries.nsmallest: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[ ]:
