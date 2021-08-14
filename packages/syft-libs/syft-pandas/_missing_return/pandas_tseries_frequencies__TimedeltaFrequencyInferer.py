#!/usr/bin/env python
# coding: utf-8

# ## pandas.tseries.frequencies._TimedeltaFrequencyInferer

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.tseries.frequencies._TimedeltaFrequencyInferer()
    return obj


# In[2]:


# pandas.tseries.frequencies._TimedeltaFrequencyInferer._infer_daily_rule
try:
    obj = class_constructor()
    ret = obj._infer_daily_rule()
    type_pandas_tseries_frequencies__TimedeltaFrequencyInferer__infer_daily_rule = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.tseries.frequencies._TimedeltaFrequencyInferer._infer_daily_rule:",
        type_pandas_tseries_frequencies__TimedeltaFrequencyInferer__infer_daily_rule,
    )
except Exception as e:
    type_pandas_tseries_frequencies__TimedeltaFrequencyInferer__infer_daily_rule = (
        "_syft_missing"
    )
    print(
        "❌ pandas.tseries.frequencies._TimedeltaFrequencyInferer._infer_daily_rule: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.tseries.frequencies._TimedeltaFrequencyInferer.month_position_check
try:
    obj = class_constructor()
    ret = obj.month_position_check()
    type_pandas_tseries_frequencies__TimedeltaFrequencyInferer_month_position_check = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.tseries.frequencies._TimedeltaFrequencyInferer.month_position_check:",
        type_pandas_tseries_frequencies__TimedeltaFrequencyInferer_month_position_check,
    )
except Exception as e:
    type_pandas_tseries_frequencies__TimedeltaFrequencyInferer_month_position_check = (
        "_syft_missing"
    )
    print(
        "❌ pandas.tseries.frequencies._TimedeltaFrequencyInferer.month_position_check: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
