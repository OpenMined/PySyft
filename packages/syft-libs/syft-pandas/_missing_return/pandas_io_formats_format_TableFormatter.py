#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.formats.format.TableFormatter

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.io.formats.format.TableFormatter()
    return obj


# In[2]:


# pandas.io.formats.format.TableFormatter.get_buffer
try:
    obj = class_constructor()
    ret = obj.get_buffer()
    type_pandas_io_formats_format_TableFormatter_get_buffer = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.formats.format.TableFormatter.get_buffer:",
        type_pandas_io_formats_format_TableFormatter_get_buffer)
except Exception as e:
    type_pandas_io_formats_format_TableFormatter_get_buffer = '_syft_missing'
    print('❌ pandas.io.formats.format.TableFormatter.get_buffer: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)


# In[3]:


# pandas.io.formats.format.TableFormatter.should_show_dimensions
try:
    obj = class_constructor()
    ret = obj.should_show_dimensions
    type_pandas_io_formats_format_TableFormatter_should_show_dimensions = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.formats.format.TableFormatter.should_show_dimensions:",
        type_pandas_io_formats_format_TableFormatter_should_show_dimensions)
except Exception as e:
    type_pandas_io_formats_format_TableFormatter_should_show_dimensions = '_syft_missing'
    print('❌ pandas.io.formats.format.TableFormatter.should_show_dimensions: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

