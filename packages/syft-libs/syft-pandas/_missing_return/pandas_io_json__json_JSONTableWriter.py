#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.json._json.JSONTableWriter

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.io.json._json.JSONTableWriter()
    return obj


# In[2]:


# pandas.io.json._json.JSONTableWriter._format_axes
try:
    obj = class_constructor()
    ret = obj._format_axes()
    type_pandas_io_json__json_JSONTableWriter__format_axes = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.json._json.JSONTableWriter._format_axes:",
        type_pandas_io_json__json_JSONTableWriter__format_axes)
except Exception as e:
    type_pandas_io_json__json_JSONTableWriter__format_axes = '_syft_missing'
    print('❌ pandas.io.json._json.JSONTableWriter._format_axes: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.io.json._json.JSONTableWriter.obj_to_write
try:
    obj = class_constructor()
    ret = obj.obj_to_write
    type_pandas_io_json__json_JSONTableWriter_obj_to_write = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.json._json.JSONTableWriter.obj_to_write:",
        type_pandas_io_json__json_JSONTableWriter_obj_to_write)
except Exception as e:
    type_pandas_io_json__json_JSONTableWriter_obj_to_write = '_syft_missing'
    print('❌ pandas.io.json._json.JSONTableWriter.obj_to_write: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)


# In[4]:


# pandas.io.json._json.JSONTableWriter.write
try:
    obj = class_constructor()
    ret = obj.write()
    type_pandas_io_json__json_JSONTableWriter_write = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.json._json.JSONTableWriter.write:",
        type_pandas_io_json__json_JSONTableWriter_write)
except Exception as e:
    type_pandas_io_json__json_JSONTableWriter_write = '_syft_missing'
    print('❌ pandas.io.json._json.JSONTableWriter.write: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

