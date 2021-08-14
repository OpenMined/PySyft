#!/usr/bin/env python
# coding: utf-8

# ## pandas._config.config.CallableDynamicDoc

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas._config.config.CallableDynamicDoc()
    return obj


# In[2]:


# pandas._config.config.CallableDynamicDoc.__call__
try:
    obj = class_constructor()
    ret = obj.__call__()
    type_pandas__config_config_CallableDynamicDoc___call__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._config.config.CallableDynamicDoc.__call__:",
        type_pandas__config_config_CallableDynamicDoc___call__,
    )
except Exception as e:
    type_pandas__config_config_CallableDynamicDoc___call__ = "_syft_missing"
    print("❌ pandas._config.config.CallableDynamicDoc.__call__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas._config.config.CallableDynamicDoc.__doc__
try:
    obj = class_constructor()
    ret = obj.__doc__
    type_pandas__config_config_CallableDynamicDoc___doc__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._config.config.CallableDynamicDoc.__doc__:",
        type_pandas__config_config_CallableDynamicDoc___doc__,
    )
except Exception as e:
    type_pandas__config_config_CallableDynamicDoc___doc__ = "_syft_missing"
    print("❌ pandas._config.config.CallableDynamicDoc.__doc__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
