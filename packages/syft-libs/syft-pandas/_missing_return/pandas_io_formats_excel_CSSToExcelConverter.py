#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.formats.excel.CSSToExcelConverter

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.io.formats.excel.CSSToExcelConverter()
    return obj


# In[2]:


# pandas.io.formats.excel.CSSToExcelConverter.build_fill
try:
    obj = class_constructor()
    ret = obj.build_fill()
    type_pandas_io_formats_excel_CSSToExcelConverter_build_fill = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.formats.excel.CSSToExcelConverter.build_fill:",
        type_pandas_io_formats_excel_CSSToExcelConverter_build_fill)
except Exception as e:
    type_pandas_io_formats_excel_CSSToExcelConverter_build_fill = '_syft_missing'
    print('❌ pandas.io.formats.excel.CSSToExcelConverter.build_fill: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

