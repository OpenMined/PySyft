#!/usr/bin/env python
# coding: utf-8

# ## pandas._libs.tslibs.parsing._timelex

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas._libs.tslibs.parsing._timelex()
    return obj


# In[2]:


# pandas._libs.tslibs.parsing._timelex.split
try:
    obj = class_constructor()
    ret = obj.split()
    type_pandas__libs_tslibs_parsing__timelex_split = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas._libs.tslibs.parsing._timelex.split:",
        type_pandas__libs_tslibs_parsing__timelex_split,
    )
except Exception as e:
    type_pandas__libs_tslibs_parsing__timelex_split = "_syft_missing"
    print("❌ pandas._libs.tslibs.parsing._timelex.split: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
