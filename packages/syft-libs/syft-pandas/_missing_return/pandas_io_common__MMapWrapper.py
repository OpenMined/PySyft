#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.common._MMapWrapper

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.io.common._MMapWrapper()
    return obj


# In[2]:


# pandas.io.common._MMapWrapper.GenericAlias
try:
    obj = class_constructor()
    ret = obj.GenericAlias()
    type_pandas_io_common__MMapWrapper_GenericAlias = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.common._MMapWrapper.GenericAlias:",
        type_pandas_io_common__MMapWrapper_GenericAlias)
except Exception as e:
    type_pandas_io_common__MMapWrapper_GenericAlias = '_syft_missing'
    print('❌ pandas.io.common._MMapWrapper.GenericAlias: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.io.common._MMapWrapper.__getattr__
try:
    obj = class_constructor()
    ret = obj.__getattr__()
    type_pandas_io_common__MMapWrapper___getattr__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.common._MMapWrapper.__getattr__:",
        type_pandas_io_common__MMapWrapper___getattr__)
except Exception as e:
    type_pandas_io_common__MMapWrapper___getattr__ = '_syft_missing'
    print('❌ pandas.io.common._MMapWrapper.__getattr__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[4]:


# pandas.io.common._MMapWrapper.__subclasshook__
try:
    obj = class_constructor()
    ret = obj.__subclasshook__()
    type_pandas_io_common__MMapWrapper___subclasshook__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.common._MMapWrapper.__subclasshook__:",
        type_pandas_io_common__MMapWrapper___subclasshook__)
except Exception as e:
    type_pandas_io_common__MMapWrapper___subclasshook__ = '_syft_missing'
    print('❌ pandas.io.common._MMapWrapper.__subclasshook__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

