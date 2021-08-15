#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.base.PandasObject

# In[1]:


# pandas.core.base.PandasObject._constructor
try:
    obj = class_constructor()
    ret = obj._constructor
    type_pandas_core_base_PandasObject__constructor = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.base.PandasObject._constructor:",
        type_pandas_core_base_PandasObject__constructor,
    )
except Exception as e:
    type_pandas_core_base_PandasObject__constructor = "_syft_missing"
    print("❌ pandas.core.base.PandasObject._constructor: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
