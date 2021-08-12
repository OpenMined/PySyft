#!/usr/bin/env python
# coding: utf-8

# ## sklearn.compose._column_transformer.make_column_selector

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.compose._column_transformer.make_column_selector()
    return obj


# In[ ]:


# sklearn.compose._column_transformer.make_column_selector.__call__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__call__()
    type_sklearn_compose__column_transformer_make_column_selector___call__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.compose._column_transformer.make_column_selector.__call__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_compose__column_transformer_make_column_selector___call__ = '_syft_missing'
    print('❌ sklearn.compose._column_transformer.make_column_selector.__call__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

