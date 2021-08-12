#!/usr/bin/env python
# coding: utf-8

# ## sklearn.base.DensityMixin

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.base.DensityMixin()
    return obj


# In[ ]:


# sklearn.base.DensityMixin.score
try:
    obj = class_constructor() # noqa F821
    ret = obj.score()
    type_sklearn_base_DensityMixin_score = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.base.DensityMixin.score: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_base_DensityMixin_score = '_syft_missing'
    print('❌ sklearn.base.DensityMixin.score: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

