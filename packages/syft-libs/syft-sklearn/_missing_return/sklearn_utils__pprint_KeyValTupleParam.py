#!/usr/bin/env python
# coding: utf-8

# ## sklearn.utils._pprint.KeyValTupleParam

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.utils._pprint.KeyValTupleParam()
    return obj


# In[ ]:


# sklearn.utils._pprint.KeyValTupleParam.__repr__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__repr__()
    type_sklearn_utils__pprint_KeyValTupleParam___repr__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.utils._pprint.KeyValTupleParam.__repr__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_utils__pprint_KeyValTupleParam___repr__ = '_syft_missing'
    print('❌ sklearn.utils._pprint.KeyValTupleParam.__repr__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

