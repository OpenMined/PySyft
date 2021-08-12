#!/usr/bin/env python
# coding: utf-8

# ## sklearn.base.ClusterMixin

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.base.ClusterMixin()
    return obj


# In[ ]:


# sklearn.base.ClusterMixin._more_tags
try:
    obj = class_constructor() # noqa F821
    ret = obj._more_tags()
    type_sklearn_base_ClusterMixin__more_tags = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.base.ClusterMixin._more_tags: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_base_ClusterMixin__more_tags = '_syft_missing'
    print('❌ sklearn.base.ClusterMixin._more_tags: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.base.ClusterMixin.fit_predict
try:
    obj = class_constructor() # noqa F821
    ret = obj.fit_predict()
    type_sklearn_base_ClusterMixin_fit_predict = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.base.ClusterMixin.fit_predict: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_base_ClusterMixin_fit_predict = '_syft_missing'
    print('❌ sklearn.base.ClusterMixin.fit_predict: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

