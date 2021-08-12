#!/usr/bin/env python
# coding: utf-8

# ## sklearn.model_selection._split.TimeSeriesSplit

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.model_selection._split.TimeSeriesSplit()
    return obj


# In[ ]:


# sklearn.model_selection._split.TimeSeriesSplit.__repr__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__repr__()
    type_sklearn_model_selection__split_TimeSeriesSplit___repr__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.model_selection._split.TimeSeriesSplit.__repr__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_model_selection__split_TimeSeriesSplit___repr__ = '_syft_missing'
    print('❌ sklearn.model_selection._split.TimeSeriesSplit.__repr__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.model_selection._split.TimeSeriesSplit._iter_test_indices
try:
    obj = class_constructor() # noqa F821
    ret = obj._iter_test_indices()
    type_sklearn_model_selection__split_TimeSeriesSplit__iter_test_indices = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.model_selection._split.TimeSeriesSplit._iter_test_indices: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_model_selection__split_TimeSeriesSplit__iter_test_indices = '_syft_missing'
    print('❌ sklearn.model_selection._split.TimeSeriesSplit._iter_test_indices: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.model_selection._split.TimeSeriesSplit._iter_test_masks
try:
    obj = class_constructor() # noqa F821
    ret = obj._iter_test_masks()
    type_sklearn_model_selection__split_TimeSeriesSplit__iter_test_masks = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.model_selection._split.TimeSeriesSplit._iter_test_masks: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_model_selection__split_TimeSeriesSplit__iter_test_masks = '_syft_missing'
    print('❌ sklearn.model_selection._split.TimeSeriesSplit._iter_test_masks: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.model_selection._split.TimeSeriesSplit.get_n_splits
try:
    obj = class_constructor() # noqa F821
    ret = obj.get_n_splits()
    type_sklearn_model_selection__split_TimeSeriesSplit_get_n_splits = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.model_selection._split.TimeSeriesSplit.get_n_splits: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_model_selection__split_TimeSeriesSplit_get_n_splits = '_syft_missing'
    print('❌ sklearn.model_selection._split.TimeSeriesSplit.get_n_splits: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.model_selection._split.TimeSeriesSplit.split
try:
    obj = class_constructor() # noqa F821
    ret = obj.split()
    type_sklearn_model_selection__split_TimeSeriesSplit_split = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.model_selection._split.TimeSeriesSplit.split: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_model_selection__split_TimeSeriesSplit_split = '_syft_missing'
    print('❌ sklearn.model_selection._split.TimeSeriesSplit.split: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

