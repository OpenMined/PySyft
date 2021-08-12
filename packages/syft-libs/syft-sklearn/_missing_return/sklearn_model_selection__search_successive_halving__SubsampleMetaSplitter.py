#!/usr/bin/env python
# coding: utf-8

# ## sklearn.model_selection._search_successive_halving._SubsampleMetaSplitter

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.model_selection._search_successive_halving._SubsampleMetaSplitter()
    return obj


# In[ ]:


# sklearn.model_selection._search_successive_halving._SubsampleMetaSplitter.split
try:
    obj = class_constructor() # noqa F821
    ret = obj.split()
    type_sklearn_model_selection__search_successive_halving__SubsampleMetaSplitter_split = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.model_selection._search_successive_halving._SubsampleMetaSplitter.split: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_model_selection__search_successive_halving__SubsampleMetaSplitter_split = '_syft_missing'
    print('❌ sklearn.model_selection._search_successive_halving._SubsampleMetaSplitter.split: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

