#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.computation.engines.NumExprEngine

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.core.computation.engines.NumExprEngine()
    return obj


# In[2]:


# pandas.core.computation.engines.NumExprEngine._evaluate
try:
    obj = class_constructor()
    ret = obj._evaluate()
    type_pandas_core_computation_engines_NumExprEngine__evaluate = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.computation.engines.NumExprEngine._evaluate:",
        type_pandas_core_computation_engines_NumExprEngine__evaluate)
except Exception as e:
    type_pandas_core_computation_engines_NumExprEngine__evaluate = '_syft_missing'
    print('❌ pandas.core.computation.engines.NumExprEngine._evaluate: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.core.computation.engines.NumExprEngine._is_aligned
try:
    obj = class_constructor()
    ret = obj._is_aligned
    type_pandas_core_computation_engines_NumExprEngine__is_aligned = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.computation.engines.NumExprEngine._is_aligned:",
        type_pandas_core_computation_engines_NumExprEngine__is_aligned)
except Exception as e:
    type_pandas_core_computation_engines_NumExprEngine__is_aligned = '_syft_missing'
    print('❌ pandas.core.computation.engines.NumExprEngine._is_aligned: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

