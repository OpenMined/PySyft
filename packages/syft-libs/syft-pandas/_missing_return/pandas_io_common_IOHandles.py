#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.common.IOHandles

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.io.common.IOHandles()
    return obj


# In[2]:


# pandas.io.common.IOHandles.__eq__
try:
    obj = class_constructor()
    ret = obj.__eq__()
    type_pandas_io_common_IOHandles___eq__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.common.IOHandles.__eq__:",
        type_pandas_io_common_IOHandles___eq__)
except Exception as e:
    type_pandas_io_common_IOHandles___eq__ = '_syft_missing'
    print('❌ pandas.io.common.IOHandles.__eq__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.io.common.IOHandles.__repr__
try:
    obj = class_constructor()
    ret = obj.__repr__()
    type_pandas_io_common_IOHandles___repr__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.common.IOHandles.__repr__:",
        type_pandas_io_common_IOHandles___repr__)
except Exception as e:
    type_pandas_io_common_IOHandles___repr__ = '_syft_missing'
    print('❌ pandas.io.common.IOHandles.__repr__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

