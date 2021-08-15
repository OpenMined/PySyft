#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.internals.array_manager.NullArrayProxy

# In[1]:


# pandas.core.internals.array_manager.NullArrayProxy.shape
try:
    obj = class_constructor()
    ret = obj.shape
    type_pandas_core_internals_array_manager_NullArrayProxy_shape = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.internals.array_manager.NullArrayProxy.shape:",
        type_pandas_core_internals_array_manager_NullArrayProxy_shape,
    )
except Exception as e:
    type_pandas_core_internals_array_manager_NullArrayProxy_shape = "_syft_missing"
    print(
        "❌ pandas.core.internals.array_manager.NullArrayProxy.shape: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
