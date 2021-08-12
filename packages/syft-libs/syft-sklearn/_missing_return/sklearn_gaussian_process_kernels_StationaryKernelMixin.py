#!/usr/bin/env python
# coding: utf-8

# ## sklearn.gaussian_process.kernels.StationaryKernelMixin

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.gaussian_process.kernels.StationaryKernelMixin()
    return obj


# In[ ]:


# sklearn.gaussian_process.kernels.StationaryKernelMixin.is_stationary
try:
    obj = class_constructor() # noqa F821
    ret = obj.is_stationary()
    type_sklearn_gaussian_process_kernels_StationaryKernelMixin_is_stationary = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.gaussian_process.kernels.StationaryKernelMixin.is_stationary: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_gaussian_process_kernels_StationaryKernelMixin_is_stationary = '_syft_missing'
    print('❌ sklearn.gaussian_process.kernels.StationaryKernelMixin.is_stationary: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

