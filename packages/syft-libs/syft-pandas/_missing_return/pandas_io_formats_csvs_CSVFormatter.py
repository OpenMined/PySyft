#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.formats.csvs.CSVFormatter

# In[1]:


import pandas
def class_constructor(*args, **kwargs):
    obj = pandas.io.formats.csvs.CSVFormatter()
    return obj


# In[2]:


# pandas.io.formats.csvs.CSVFormatter._save_header
try:
    obj = class_constructor()
    ret = obj._save_header()
    type_pandas_io_formats_csvs_CSVFormatter__save_header = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.formats.csvs.CSVFormatter._save_header:",
        type_pandas_io_formats_csvs_CSVFormatter__save_header)
except Exception as e:
    type_pandas_io_formats_csvs_CSVFormatter__save_header = '_syft_missing'
    print('❌ pandas.io.formats.csvs.CSVFormatter._save_header: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

