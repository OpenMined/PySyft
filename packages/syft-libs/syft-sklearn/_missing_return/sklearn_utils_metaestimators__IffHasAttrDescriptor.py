#!/usr/bin/env python
# coding: utf-8

# ## sklearn.utils.metaestimators._IffHasAttrDescriptor

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.utils.metaestimators._IffHasAttrDescriptor()
    return obj


# In[ ]:


# sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__
try:
    obj = class_constructor() # noqa F821
    ret = obj.__get__()
    type_sklearn_utils_metaestimators__IffHasAttrDescriptor___get__ = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_utils_metaestimators__IffHasAttrDescriptor___get__ = '_syft_missing'
    print('❌ sklearn.utils.metaestimators._IffHasAttrDescriptor.__get__: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

