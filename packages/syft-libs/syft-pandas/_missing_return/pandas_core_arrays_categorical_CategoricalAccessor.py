#!/usr/bin/env python
# coding: utf-8

# ## pandas.core.arrays.categorical.CategoricalAccessor

# In[1]:


import pandas
import pandas as pd
def class_constructor(*args, **kwargs):
    obj = pandas.core.arrays.categorical.CategoricalAccessor(pd.Series([1,2,1.1]).astype("category"))
    return obj


# In[2]:


class_constructor()


# In[3]:


# pandas.core.arrays.categorical.CategoricalAccessor.__dir__
try:
    obj = class_constructor()
    ret = obj.__dir__()
    type_pandas_core_arrays_categorical_CategoricalAccessor___dir__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.__dir__:",
        type_pandas_core_arrays_categorical_CategoricalAccessor___dir__)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor___dir__ = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.__dir__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[4]:


# pandas.core.arrays.categorical.CategoricalAccessor.__setattr__
try:
    obj = class_constructor()
    print(obj.__dict__)
    ret = obj.__setattr__("codes",[1,2,3])
    type_pandas_core_arrays_categorical_CategoricalAccessor___setattr__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.__setattr__:",
        type_pandas_core_arrays_categorical_CategoricalAccessor___setattr__)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor___setattr__ = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.__setattr__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[5]:


# pandas.core.arrays.categorical.CategoricalAccessor.__sizeof__
try:
    obj = class_constructor()
    ret = obj.__sizeof__()
    type_pandas_core_arrays_categorical_CategoricalAccessor___sizeof__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.__sizeof__:",
        type_pandas_core_arrays_categorical_CategoricalAccessor___sizeof__)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor___sizeof__ = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.__sizeof__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[6]:


# pandas.core.arrays.categorical.CategoricalAccessor._add_delegate_accessors
try:
    obj = class_constructor()
    ret = obj._add_delegate_accessors()
    type_pandas_core_arrays_categorical_CategoricalAccessor__add_delegate_accessors = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._add_delegate_accessors:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__add_delegate_accessors)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__add_delegate_accessors = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._add_delegate_accessors: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[7]:


# pandas.core.arrays.categorical.CategoricalAccessor._constructor
try:
    obj = class_constructor()
    ret = obj._constructor
    type_pandas_core_arrays_categorical_CategoricalAccessor__constructor = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._constructor:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__constructor)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__constructor = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._constructor: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[8]:


# pandas.core.arrays.categorical.CategoricalAccessor._delegate_method
try:
    obj = class_constructor()
    ret = obj._delegate_method()
    type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_method = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._delegate_method:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_method)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_method = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._delegate_method: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[9]:


# pandas.core.arrays.categorical.CategoricalAccessor._delegate_property_get
try:
    obj = class_constructor()
    ret = obj._delegate_property_get()
    type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_property_get = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._delegate_property_get:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_property_get)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_property_get = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._delegate_property_get: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[10]:


# pandas.core.arrays.categorical.CategoricalAccessor._delegate_property_set
try:
    obj = class_constructor()
    ret = obj._delegate_property_set("categories",[1,2,11])
    type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_property_set = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._delegate_property_set:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_property_set)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__delegate_property_set = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._delegate_property_set: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[11]:


# pandas.core.arrays.categorical.CategoricalAccessor._dir_additions
try:
    obj = class_constructor()
    ret = obj._dir_additions()
    type_pandas_core_arrays_categorical_CategoricalAccessor__dir_additions = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._dir_additions:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__dir_additions)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__dir_additions = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._dir_additions: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[12]:


# pandas.core.arrays.categorical.CategoricalAccessor._dir_deletions
try:
    obj = class_constructor()
    ret = obj._dir_deletions()
    type_pandas_core_arrays_categorical_CategoricalAccessor__dir_deletions = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._dir_deletions:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__dir_deletions)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__dir_deletions = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._dir_deletions: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[13]:


# pandas.core.arrays.categorical.CategoricalAccessor._freeze
try:
    obj = class_constructor()
    ret = obj._freeze()
    type_pandas_core_arrays_categorical_CategoricalAccessor__freeze = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._freeze:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__freeze)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__freeze = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._freeze: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[14]:


# pandas.core.arrays.categorical.CategoricalAccessor._validate
try:
    obj = class_constructor()
    ret = obj._validate(pd.Series([1,11,12]).astype("category"))
    type_pandas_core_arrays_categorical_CategoricalAccessor__validate = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor._validate:",
        type_pandas_core_arrays_categorical_CategoricalAccessor__validate)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor__validate = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor._validate: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[15]:


# pandas.core.arrays.categorical.CategoricalAccessor.add_categories
try:
    obj = class_constructor()
    ret = obj.add_categories("c")
    type_pandas_core_arrays_categorical_CategoricalAccessor_add_categories = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.add_categories:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_add_categories)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_add_categories = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.add_categories: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[16]:


# pandas.core.arrays.categorical.CategoricalAccessor.as_ordered
try:
    obj = class_constructor()
    ret = obj.as_ordered()
    type_pandas_core_arrays_categorical_CategoricalAccessor_as_ordered = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.as_ordered:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_as_ordered)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_as_ordered = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.as_ordered: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[17]:


# pandas.core.arrays.categorical.CategoricalAccessor.as_unordered
try:
    obj = class_constructor()
    ret = obj.as_unordered()
    type_pandas_core_arrays_categorical_CategoricalAccessor_as_unordered = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.as_unordered:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_as_unordered)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_as_unordered = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.as_unordered: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[18]:


# pandas.core.arrays.categorical.CategoricalAccessor.categories
try:
    obj = class_constructor()
    ret = obj.categories
    type_pandas_core_arrays_categorical_CategoricalAccessor_categories = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.categories:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_categories)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_categories = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.categories: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[19]:


# pandas.core.arrays.categorical.CategoricalAccessor.codes
try:
    obj = class_constructor()
    ret = obj.codes
    type_pandas_core_arrays_categorical_CategoricalAccessor_codes = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.codes:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_codes)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_codes = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.codes: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[20]:


# pandas.core.arrays.categorical.CategoricalAccessor.ordered
try:
    obj = class_constructor()
    ret = obj.ordered
    type_pandas_core_arrays_categorical_CategoricalAccessor_ordered = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.ordered:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_ordered)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_ordered = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.ordered: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[21]:


# pandas.core.arrays.categorical.CategoricalAccessor.remove_categories
try:
    obj = class_constructor()
    ret = obj.remove_categories(1)
    type_pandas_core_arrays_categorical_CategoricalAccessor_remove_categories = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.remove_categories:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_remove_categories)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_remove_categories = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.remove_categories: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[22]:


# pandas.core.arrays.categorical.CategoricalAccessor.remove_unused_categories
try:
    obj = class_constructor()
    ret = obj.remove_unused_categories()
    type_pandas_core_arrays_categorical_CategoricalAccessor_remove_unused_categories = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.remove_unused_categories:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_remove_unused_categories)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_remove_unused_categories = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.remove_unused_categories: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[23]:


# pandas.core.arrays.categorical.CategoricalAccessor.rename_categories
try:
    obj = class_constructor()
    ret = obj.rename_categories([1.1,3.0,11])
    type_pandas_core_arrays_categorical_CategoricalAccessor_rename_categories = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.rename_categories:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_rename_categories)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_rename_categories = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.rename_categories: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[24]:


# pandas.core.arrays.categorical.CategoricalAccessor.reorder_categories
try:
    obj = class_constructor()
    ret = obj.reorder_categories([1.1,1,2.0])
    type_pandas_core_arrays_categorical_CategoricalAccessor_reorder_categories = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.reorder_categories:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_reorder_categories)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_reorder_categories = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.reorder_categories: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[25]:


# pandas.core.arrays.categorical.CategoricalAccessor.set_categories
try:
    obj = class_constructor()
    ret = obj.set_categories([1,11,12])
    type_pandas_core_arrays_categorical_CategoricalAccessor_set_categories = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.core.arrays.categorical.CategoricalAccessor.set_categories:",
        type_pandas_core_arrays_categorical_CategoricalAccessor_set_categories)
except Exception as e:
    type_pandas_core_arrays_categorical_CategoricalAccessor_set_categories = '_syft_missing'
    print('❌ pandas.core.arrays.categorical.CategoricalAccessor.set_categories: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[ ]:




