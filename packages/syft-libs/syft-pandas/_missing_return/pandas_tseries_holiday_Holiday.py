#!/usr/bin/env python
# coding: utf-8

# ## pandas.tseries.holiday.Holiday

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.tseries.holiday.Holiday()
    return obj


# In[2]:


# pandas.tseries.holiday.Holiday._apply_rule
try:
    obj = class_constructor()
    ret = obj._apply_rule()
    type_pandas_tseries_holiday_Holiday__apply_rule = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.tseries.holiday.Holiday._apply_rule:",
        type_pandas_tseries_holiday_Holiday__apply_rule)
except Exception as e:
    type_pandas_tseries_holiday_Holiday__apply_rule = '_syft_missing'
    print('❌ pandas.tseries.holiday.Holiday._apply_rule: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.tseries.holiday.Holiday._reference_dates
try:
    obj = class_constructor()
    ret = obj._reference_dates()
    type_pandas_tseries_holiday_Holiday__reference_dates = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.tseries.holiday.Holiday._reference_dates:",
        type_pandas_tseries_holiday_Holiday__reference_dates)
except Exception as e:
    type_pandas_tseries_holiday_Holiday__reference_dates = '_syft_missing'
    print('❌ pandas.tseries.holiday.Holiday._reference_dates: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[4]:


# pandas.tseries.holiday.Holiday.dates
try:
    obj = class_constructor()
    ret = obj.dates()
    type_pandas_tseries_holiday_Holiday_dates = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.tseries.holiday.Holiday.dates:",
        type_pandas_tseries_holiday_Holiday_dates)
except Exception as e:
    type_pandas_tseries_holiday_Holiday_dates = '_syft_missing'
    print('❌ pandas.tseries.holiday.Holiday.dates: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

