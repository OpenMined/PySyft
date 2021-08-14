#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.base.NoNewAttributesMixin

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.base.NoNewAttributesMixin()
    return obj


# In[2]:


# pandas.core.base.NoNewAttributesMixin.__setattr__
try:
    obj = class_constructor()
    ret = obj.__setattr__()
    type_pandas_core_base_NoNewAttributesMixin___setattr__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.base.NoNewAttributesMixin.__setattr__:",
        type_pandas_core_base_NoNewAttributesMixin___setattr__,
    )
except Exception as e:
    type_pandas_core_base_NoNewAttributesMixin___setattr__ = "_syft_missing"
    print("❌ pandas.core.base.NoNewAttributesMixin.__setattr__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.base.NoNewAttributesMixin._freeze
try:
    obj = class_constructor()
    ret = obj._freeze()
    type_pandas_core_base_NoNewAttributesMixin__freeze = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.base.NoNewAttributesMixin._freeze:",
        type_pandas_core_base_NoNewAttributesMixin__freeze,
    )
except Exception as e:
    type_pandas_core_base_NoNewAttributesMixin__freeze = "_syft_missing"
    print("❌ pandas.core.base.NoNewAttributesMixin._freeze: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
