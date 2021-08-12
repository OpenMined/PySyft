#!/usr/bin/env python
# coding: utf-8

# ## sklearn.base.OutlierMixin

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.base.OutlierMixin()
    return obj


# In[ ]:


# sklearn.base.OutlierMixin.fit_predict
try:
    obj = class_constructor() # noqa F821
    ret = obj.fit_predict()
    type_sklearn_base_OutlierMixin_fit_predict = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.base.OutlierMixin.fit_predict: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_base_OutlierMixin_fit_predict = '_syft_missing'
    print('❌ sklearn.base.OutlierMixin.fit_predict: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

