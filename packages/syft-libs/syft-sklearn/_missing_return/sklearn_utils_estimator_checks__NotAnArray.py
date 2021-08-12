#!/usr/bin/env python
# coding: utf-8

# ## sklearn.utils.estimator_checks._NotAnArray

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.utils.estimator_checks._NotAnArray()
    return obj


# In[ ]:


# sklearn.utils.estimator_checks._NotAnArray.__array__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__array__()
    type_sklearn_utils_estimator_checks__NotAnArray___array__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.utils.estimator_checks._NotAnArray.__array__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_utils_estimator_checks__NotAnArray___array__ = '_syft_missing'
    print('❌ sklearn.utils.estimator_checks._NotAnArray.__array__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.utils.estimator_checks._NotAnArray.__array_function__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__array_function__()
    type_sklearn_utils_estimator_checks__NotAnArray___array_function__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.utils.estimator_checks._NotAnArray.__array_function__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_utils_estimator_checks__NotAnArray___array_function__ = '_syft_missing'
    print('❌ sklearn.utils.estimator_checks._NotAnArray.__array_function__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

