#!/usr/bin/env python
# coding: utf-8

# ## sklearn.cluster._birch._CFSubcluster

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.cluster._birch._CFSubcluster()
    return obj


# In[ ]:


# sklearn.cluster._birch._CFSubcluster.merge_subcluster
try:
    obj = class_constructor() # noqa F821
    ret = obj.merge_subcluster()
    type_sklearn_cluster__birch__CFSubcluster_merge_subcluster = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.cluster._birch._CFSubcluster.merge_subcluster: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_cluster__birch__CFSubcluster_merge_subcluster = '_syft_missing'
    print('❌ sklearn.cluster._birch._CFSubcluster.merge_subcluster: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.cluster._birch._CFSubcluster.radius
try:
    obj = class_constructor()
    ret = obj.radius
    type_sklearn_cluster__birch__CFSubcluster_radius = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.cluster._birch._CFSubcluster.radius:', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_cluster__birch__CFSubcluster_radius = '_syft_missing'
    print('❌ sklearn.cluster._birch._CFSubcluster.radius: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.cluster._birch._CFSubcluster.update
try:
    obj = class_constructor() # noqa F821
    ret = obj.update()
    type_sklearn_cluster__birch__CFSubcluster_update = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.cluster._birch._CFSubcluster.update: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_cluster__birch__CFSubcluster_update = '_syft_missing'
    print('❌ sklearn.cluster._birch._CFSubcluster.update: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

