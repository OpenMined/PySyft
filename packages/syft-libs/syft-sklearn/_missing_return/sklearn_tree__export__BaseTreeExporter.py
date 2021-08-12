#!/usr/bin/env python
# coding: utf-8

# ## sklearn.tree._export._BaseTreeExporter

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.tree._export._BaseTreeExporter()
    return obj


# In[ ]:


# sklearn.tree._export._BaseTreeExporter.get_color
try:
    obj = class_constructor() # noqa F821
    ret = obj.get_color()
    type_sklearn_tree__export__BaseTreeExporter_get_color = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.tree._export._BaseTreeExporter.get_color: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_tree__export__BaseTreeExporter_get_color = '_syft_missing'
    print('❌ sklearn.tree._export._BaseTreeExporter.get_color: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.tree._export._BaseTreeExporter.get_fill_color
try:
    obj = class_constructor() # noqa F821
    ret = obj.get_fill_color()
    type_sklearn_tree__export__BaseTreeExporter_get_fill_color = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.tree._export._BaseTreeExporter.get_fill_color: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_tree__export__BaseTreeExporter_get_fill_color = '_syft_missing'
    print('❌ sklearn.tree._export._BaseTreeExporter.get_fill_color: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[ ]:


# sklearn.tree._export._BaseTreeExporter.node_to_str
try:
    obj = class_constructor() # noqa F821
    ret = obj.node_to_str()
    type_sklearn_tree__export__BaseTreeExporter_node_to_str = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.tree._export._BaseTreeExporter.node_to_str: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_tree__export__BaseTreeExporter_node_to_str = '_syft_missing'
    print('❌ sklearn.tree._export._BaseTreeExporter.node_to_str: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

