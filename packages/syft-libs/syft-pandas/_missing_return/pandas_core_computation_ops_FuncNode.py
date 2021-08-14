#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.computation.ops.FuncNode

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.computation.ops.FuncNode()
    return obj


# In[2]:


# pandas.core.computation.ops.FuncNode.__call__
try:
    obj = class_constructor()
    ret = obj.__call__()
    type_pandas_core_computation_ops_FuncNode___call__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.computation.ops.FuncNode.__call__:",
        type_pandas_core_computation_ops_FuncNode___call__)
except Exception as e:
    type_pandas_core_computation_ops_FuncNode___call__ = '_syft_missing'
    print('❌ pandas.core.computation.ops.FuncNode.__call__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

