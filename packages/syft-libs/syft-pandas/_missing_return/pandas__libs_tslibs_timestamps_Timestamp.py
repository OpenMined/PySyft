#!/usr/bin/env python
# coding: utf-8

# ## pandas._libs.tslibs.timestamps.Timestamp

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas._libs.tslibs.timestamps.Timestamp(2020, 3, 4)
    return obj


# In[2]:


class_constructor()


# In[3]:


# stdlib
# pandas._libs.tslibs.timestamps.Timestamp.combine
from datetime import date
from datetime import time

try:
    obj = class_constructor()
    ret = obj.combine(date(2020, 3, 14), time(15, 30, 15))
    type_pandas__libs_tslibs_timestamps_Timestamp_combine = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.combine:",
        type_pandas__libs_tslibs_timestamps_Timestamp_combine,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_combine = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.combine: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas._libs.tslibs.timestamps.Timestamp.freqstr
try:
    obj = class_constructor()
    ret = obj.freqstr
    type_pandas__libs_tslibs_timestamps_Timestamp_freqstr = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.freqstr:",
        type_pandas__libs_tslibs_timestamps_Timestamp_freqstr,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_freqstr = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.freqstr: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[5]:


# pandas._libs.tslibs.timestamps.Timestamp.fromordinal
try:
    obj = class_constructor()
    ret = obj.fromordinal(737425)
    type_pandas__libs_tslibs_timestamps_Timestamp_fromordinal = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.fromordinal:",
        type_pandas__libs_tslibs_timestamps_Timestamp_fromordinal,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_fromordinal = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.fromordinal: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[6]:


# pandas._libs.tslibs.timestamps.Timestamp.fromtimestamp
try:
    obj = class_constructor()
    ret = obj.fromtimestamp(1584199972)
    type_pandas__libs_tslibs_timestamps_Timestamp_fromtimestamp = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.fromtimestamp:",
        type_pandas__libs_tslibs_timestamps_Timestamp_fromtimestamp,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_fromtimestamp = "_syft_missing"
    print(
        "❌ pandas._libs.tslibs.timestamps.Timestamp.fromtimestamp: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[7]:


# pandas._libs.tslibs.timestamps.Timestamp.now
try:
    obj = class_constructor()
    ret = obj.now()
    type_pandas__libs_tslibs_timestamps_Timestamp_now = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.now:",
        type_pandas__libs_tslibs_timestamps_Timestamp_now,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_now = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.now: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[8]:


# pandas._libs.tslibs.timestamps.Timestamp.strptime

try:
    obj = class_constructor()
    ret = obj.strptime("10/10/20", "dd/mm/yy")
    type_pandas__libs_tslibs_timestamps_Timestamp_strptime = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.strptime:",
        type_pandas__libs_tslibs_timestamps_Timestamp_strptime,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_strptime = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.strptime: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[9]:


# pandas._libs.tslibs.timestamps.Timestamp.today
try:
    obj = class_constructor()
    ret = obj.today()
    type_pandas__libs_tslibs_timestamps_Timestamp_today = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.today:",
        type_pandas__libs_tslibs_timestamps_Timestamp_today,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_today = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.today: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[10]:


# pandas._libs.tslibs.timestamps.Timestamp.tz
try:
    obj = class_constructor()
    ret = obj.tz
    type_pandas__libs_tslibs_timestamps_Timestamp_tz = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.tz:",
        type_pandas__libs_tslibs_timestamps_Timestamp_tz,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_tz = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.tz: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[11]:


# pandas._libs.tslibs.timestamps.Timestamp.utcfromtimestamp
try:
    obj = class_constructor()
    ret = obj.utcfromtimestamp(1584199972)
    type_pandas__libs_tslibs_timestamps_Timestamp_utcfromtimestamp = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.utcfromtimestamp:",
        type_pandas__libs_tslibs_timestamps_Timestamp_utcfromtimestamp,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_utcfromtimestamp = "_syft_missing"
    print(
        "❌ pandas._libs.tslibs.timestamps.Timestamp.utcfromtimestamp: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[12]:


# pandas._libs.tslibs.timestamps.Timestamp.utcnow
try:
    obj = class_constructor()
    ret = obj.utcnow()
    type_pandas__libs_tslibs_timestamps_Timestamp_utcnow = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.timestamps.Timestamp.utcnow:",
        type_pandas__libs_tslibs_timestamps_Timestamp_utcnow,
    )
except Exception as e:
    type_pandas__libs_tslibs_timestamps_Timestamp_utcnow = "_syft_missing"
    print("❌ pandas._libs.tslibs.timestamps.Timestamp.utcnow: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
