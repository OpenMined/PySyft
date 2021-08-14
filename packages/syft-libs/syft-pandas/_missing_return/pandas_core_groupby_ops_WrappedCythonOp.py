#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.groupby.ops.WrappedCythonOp

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.core.groupby.ops.WrappedCythonOp()
    return obj


# In[2]:


# pandas.core.groupby.ops.WrappedCythonOp._disallow_invalid_ops
try:
    obj = class_constructor()
    ret = obj._disallow_invalid_ops()
    type_pandas_core_groupby_ops_WrappedCythonOp__disallow_invalid_ops = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.groupby.ops.WrappedCythonOp._disallow_invalid_ops:",
        type_pandas_core_groupby_ops_WrappedCythonOp__disallow_invalid_ops,
    )
except Exception as e:
    type_pandas_core_groupby_ops_WrappedCythonOp__disallow_invalid_ops = "_syft_missing"
    print(
        "❌ pandas.core.groupby.ops.WrappedCythonOp._disallow_invalid_ops: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.core.groupby.ops.WrappedCythonOp._get_cython_function
try:
    obj = class_constructor()
    ret = obj._get_cython_function()
    type_pandas_core_groupby_ops_WrappedCythonOp__get_cython_function = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.groupby.ops.WrappedCythonOp._get_cython_function:",
        type_pandas_core_groupby_ops_WrappedCythonOp__get_cython_function,
    )
except Exception as e:
    type_pandas_core_groupby_ops_WrappedCythonOp__get_cython_function = "_syft_missing"
    print(
        "❌ pandas.core.groupby.ops.WrappedCythonOp._get_cython_function: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[4]:


# pandas.core.groupby.ops.WrappedCythonOp.get_cython_func_and_vals
try:
    obj = class_constructor()
    ret = obj.get_cython_func_and_vals()
    type_pandas_core_groupby_ops_WrappedCythonOp_get_cython_func_and_vals = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.core.groupby.ops.WrappedCythonOp.get_cython_func_and_vals:",
        type_pandas_core_groupby_ops_WrappedCythonOp_get_cython_func_and_vals,
    )
except Exception as e:
    type_pandas_core_groupby_ops_WrappedCythonOp_get_cython_func_and_vals = (
        "_syft_missing"
    )
    print(
        "❌ pandas.core.groupby.ops.WrappedCythonOp.get_cython_func_and_vals: Return unavailable"
    )
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
