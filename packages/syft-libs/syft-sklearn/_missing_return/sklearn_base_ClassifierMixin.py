#!/usr/bin/env python
# coding: utf-8

# ## sklearn.base.ClassifierMixin

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.base.ClassifierMixin()
    return obj


# In[ ]:


# sklearn.base.ClassifierMixin._more_tags
try:
    obj = class_constructor() # noqa F821
    ret = obj._more_tags()
    type_sklearn_base_ClassifierMixin__more_tags = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.base.ClassifierMixin._more_tags: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_base_ClassifierMixin__more_tags = '_syft_missing'
    print('❌ sklearn.base.ClassifierMixin._more_tags: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.base.ClassifierMixin.score
try:
    obj = class_constructor() # noqa F821
    ret = obj.score()
    type_sklearn_base_ClassifierMixin_score = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.base.ClassifierMixin.score: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_base_ClassifierMixin_score = '_syft_missing'
    print('❌ sklearn.base.ClassifierMixin.score: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

