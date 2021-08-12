#!/usr/bin/env python
# coding: utf-8

# ## sklearn.utils._estimator_html_repr._VisualBlock

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.utils._estimator_html_repr._VisualBlock()
    return obj


# In[ ]:


# sklearn.utils._estimator_html_repr._VisualBlock._sk_visual_block_
try:
    obj = class_constructor() # noqa F821
    ret = obj._sk_visual_block_()
    type_sklearn_utils__estimator_html_repr__VisualBlock__sk_visual_block_ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.utils._estimator_html_repr._VisualBlock._sk_visual_block_: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_utils__estimator_html_repr__VisualBlock__sk_visual_block_ = '_syft_missing'
    print('❌ sklearn.utils._estimator_html_repr._VisualBlock._sk_visual_block_: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

