#!/usr/bin/env python
# coding: utf-8

# ## sklearn.gaussian_process.kernels.NormalizedKernelMixin

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.gaussian_process.kernels.NormalizedKernelMixin()
    return obj


# In[ ]:


# sklearn.gaussian_process.kernels.NormalizedKernelMixin.diag
try:
    obj = class_constructor() # noqa F821
    ret = obj.diag()
    type_sklearn_gaussian_process_kernels_NormalizedKernelMixin_diag = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.gaussian_process.kernels.NormalizedKernelMixin.diag: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_gaussian_process_kernels_NormalizedKernelMixin_diag = '_syft_missing'
    print('❌ sklearn.gaussian_process.kernels.NormalizedKernelMixin.diag: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

