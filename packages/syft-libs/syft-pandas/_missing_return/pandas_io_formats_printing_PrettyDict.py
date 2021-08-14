#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.formats.printing.PrettyDict

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.io.formats.printing.PrettyDict()
    return obj


# In[2]:


# pandas.io.formats.printing.PrettyDict.__init_subclass__
try:
    obj = class_constructor()
    ret = obj.__init_subclass__()
    type_pandas_io_formats_printing_PrettyDict___init_subclass__ = (
    getattr(ret, '__module__', None) + '.' + ret.__class__.__name__
        if getattr(ret, '__module__', None)
        else ret.__class__.__name__
        )
    print("✅ pandas.io.formats.printing.PrettyDict.__init_subclass__:",
        type_pandas_io_formats_printing_PrettyDict___init_subclass__)
except Exception as e:
    type_pandas_io_formats_printing_PrettyDict___init_subclass__ = '_syft_missing'
    print('❌ pandas.io.formats.printing.PrettyDict.__init_subclass__: Return unavailable')
    print("  Please fix this return type code until there is no exception")
    print('   Error:', e)

