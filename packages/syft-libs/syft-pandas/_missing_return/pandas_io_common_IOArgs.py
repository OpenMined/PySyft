#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.common.IOArgs

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.io.common.IOArgs()
    return obj


# In[2]:


# pandas.io.common.IOArgs.__eq__
try:
    obj = class_constructor()
    ret = obj.__eq__()
    type_pandas_io_common_IOArgs___eq__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.common.IOArgs.__eq__:",
        type_pandas_io_common_IOArgs___eq__)
except Exception as e:
    type_pandas_io_common_IOArgs___eq__ = '_syft_missing'
    print('❌ pandas.io.common.IOArgs.__eq__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.io.common.IOArgs.__repr__
try:
    obj = class_constructor()
    ret = obj.__repr__()
    type_pandas_io_common_IOArgs___repr__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.common.IOArgs.__repr__:",
        type_pandas_io_common_IOArgs___repr__)
except Exception as e:
    type_pandas_io_common_IOArgs___repr__ = '_syft_missing'
    print('❌ pandas.io.common.IOArgs.__repr__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

