#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.json._json.SeriesWriter

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.io.json._json.SeriesWriter()
    return obj


# In[2]:


# pandas.io.json._json.SeriesWriter._format_axes
try:
    obj = class_constructor()
    ret = obj._format_axes()
    type_pandas_io_json__json_SeriesWriter__format_axes = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.json._json.SeriesWriter._format_axes:",
        type_pandas_io_json__json_SeriesWriter__format_axes)
except Exception as e:
    type_pandas_io_json__json_SeriesWriter__format_axes = '_syft_missing'
    print('❌ pandas.io.json._json.SeriesWriter._format_axes: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.io.json._json.SeriesWriter._write
try:
    obj = class_constructor()
    ret = obj._write()
    type_pandas_io_json__json_SeriesWriter__write = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.json._json.SeriesWriter._write:",
        type_pandas_io_json__json_SeriesWriter__write)
except Exception as e:
    type_pandas_io_json__json_SeriesWriter__write = '_syft_missing'
    print('❌ pandas.io.json._json.SeriesWriter._write: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[4]:


# pandas.io.json._json.SeriesWriter.write
try:
    obj = class_constructor()
    ret = obj.write()
    type_pandas_io_json__json_SeriesWriter_write = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.json._json.SeriesWriter.write:",
        type_pandas_io_json__json_SeriesWriter_write)
except Exception as e:
    type_pandas_io_json__json_SeriesWriter_write = '_syft_missing'
    print('❌ pandas.io.json._json.SeriesWriter.write: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

