#!/usr/bin/env python
# coding: utf-8

# ## sklearn.metrics._scorer._MultimetricScorer

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.metrics._scorer._MultimetricScorer()
    return obj


# In[ ]:


# sklearn.metrics._scorer._MultimetricScorer.__call__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__call__()
    type_sklearn_metrics__scorer__MultimetricScorer___call__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.metrics._scorer._MultimetricScorer.__call__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_metrics__scorer__MultimetricScorer___call__ = '_syft_missing'
    print('❌ sklearn.metrics._scorer._MultimetricScorer.__call__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.metrics._scorer._MultimetricScorer._use_cache
try:
    obj = class_constructor() # noqa F821
    ret = obj._use_cache()
    type_sklearn_metrics__scorer__MultimetricScorer__use_cache = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.metrics._scorer._MultimetricScorer._use_cache: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_metrics__scorer__MultimetricScorer__use_cache = '_syft_missing'
    print('❌ sklearn.metrics._scorer._MultimetricScorer._use_cache: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

