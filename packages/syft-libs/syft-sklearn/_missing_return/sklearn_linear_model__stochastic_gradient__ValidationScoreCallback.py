#!/usr/bin/env python
# coding: utf-8

# ## sklearn.linear_model._stochastic_gradient._ValidationScoreCallback

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.linear_model._stochastic_gradient._ValidationScoreCallback()
    return obj


# In[ ]:


# sklearn.linear_model._stochastic_gradient._ValidationScoreCallback.__call__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__call__()
    type_sklearn_linear_model__stochastic_gradient__ValidationScoreCallback___call__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.linear_model._stochastic_gradient._ValidationScoreCallback.__call__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_linear_model__stochastic_gradient__ValidationScoreCallback___call__ = '_syft_missing'
    print('❌ sklearn.linear_model._stochastic_gradient._ValidationScoreCallback.__call__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

