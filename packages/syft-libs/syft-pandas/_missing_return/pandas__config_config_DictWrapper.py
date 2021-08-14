#!/usr/bin/env python
# coding: utf-8

# ## pandas._config.config.DictWrapper

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas._config.config.DictWrapper({"abc": 11})
    return obj


# In[2]:


# pandas._config.config.DictWrapper.__getattr__
try:
    obj = class_constructor()
    ret = obj.__getattr__("abc")
    type_pandas__config_config_DictWrapper___getattr__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._config.config.DictWrapper.__getattr__:",
        type_pandas__config_config_DictWrapper___getattr__,
    )
except Exception as e:
    type_pandas__config_config_DictWrapper___getattr__ = "_syft_missing"
    print("❌ pandas._config.config.DictWrapper.__getattr__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[ ]:
