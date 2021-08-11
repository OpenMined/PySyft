#!/usr/bin/env python
# coding: utf-8

# ## pandas.tseries.holiday.HolidayCalendarMetaClass

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.tseries.holiday.HolidayCalendarMetaClass()
    return obj


# In[2]:


# pandas.tseries.holiday.HolidayCalendarMetaClass.__new__
try:
    obj = class_constructor()
    ret = obj.__new__()
    type_pandas_tseries_holiday_HolidayCalendarMetaClass___new__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.tseries.holiday.HolidayCalendarMetaClass.__new__:",
        type_pandas_tseries_holiday_HolidayCalendarMetaClass___new__)
except Exception as e:
    type_pandas_tseries_holiday_HolidayCalendarMetaClass___new__ = '_syft_missing'
    print('❌ pandas.tseries.holiday.HolidayCalendarMetaClass.__new__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

