#!/usr/bin/env python
# coding: utf-8

# ## pandas._config.config.option_context
#
# https://github.com/pandas-dev/pandas/blob/77443dce2734d57484e3f5f38eba6d1897089182/pandas/_config/config.py#L392

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas._config.config.option_context("mode.chained_assignment", 101)
    return obj


# In[2]:


# pandas._config.config.option_context.__call__
try:
    obj = class_constructor()
    ret = obj.__call__()
    type_pandas__config_config_option_context___call__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._config.config.option_context.__call__:",
        type_pandas__config_config_option_context___call__,
    )
except Exception as e:
    type_pandas__config_config_option_context___call__ = "_syft_missing"
    print("❌ pandas._config.config.option_context.__call__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas._config.config.option_context.__enter__
try:
    obj = class_constructor()
    ret = obj.__enter__()
    type_pandas__config_config_option_context___enter__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._config.config.option_context.__enter__:",
        type_pandas__config_config_option_context___enter__,
    )
except Exception as e:
    type_pandas__config_config_option_context___enter__ = "_syft_missing"
    print("❌ pandas._config.config.option_context.__enter__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas._config.config.option_context.__exit__
try:
    obj = class_constructor()
    ret = obj.__exit__()
    type_pandas__config_config_option_context___exit__ = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._config.config.option_context.__exit__:",
        type_pandas__config_config_option_context___exit__,
    )
except Exception as e:
    type_pandas__config_config_option_context___exit__ = "_syft_missing"
    print("❌ pandas._config.config.option_context.__exit__: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[5]:


# pandas._config.config.option_context._recreate_cm
try:
    obj = class_constructor()
    ret = obj._recreate_cm()
    type_pandas__config_config_option_context__recreate_cm = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._config.config.option_context._recreate_cm:",
        type_pandas__config_config_option_context__recreate_cm,
    )
except Exception as e:
    type_pandas__config_config_option_context__recreate_cm = "_syft_missing"
    print("❌ pandas._config.config.option_context._recreate_cm: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[ ]:
