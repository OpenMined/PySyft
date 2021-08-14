#!/usr/bin/env python
# coding: utf-8

# ## pandas.io.json._json.FrameWriter

# In[1]:


# third party
import pandas


def class_constructor(*args, **kwargs):
    obj = pandas.io.json._json.FrameWriter()
    return obj


# In[2]:


# pandas.io.json._json.FrameWriter._format_axes
try:
    obj = class_constructor()
    ret = obj._format_axes()
    type_pandas_io_json__json_FrameWriter__format_axes = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.io.json._json.FrameWriter._format_axes:",
        type_pandas_io_json__json_FrameWriter__format_axes,
    )
except Exception as e:
    type_pandas_io_json__json_FrameWriter__format_axes = "_syft_missing"
    print("❌ pandas.io.json._json.FrameWriter._format_axes: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)


# In[3]:


# pandas.io.json._json.FrameWriter.obj_to_write
try:
    obj = class_constructor()
    ret = obj.obj_to_write
    type_pandas_io_json__json_FrameWriter_obj_to_write = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.io.json._json.FrameWriter.obj_to_write:",
        type_pandas_io_json__json_FrameWriter_obj_to_write,
    )
except Exception as e:
    type_pandas_io_json__json_FrameWriter_obj_to_write = "_syft_missing"
    print("❌ pandas.io.json._json.FrameWriter.obj_to_write: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[4]:


# pandas.io.json._json.FrameWriter.write
try:
    obj = class_constructor()
    ret = obj.write()
    type_pandas_io_json__json_FrameWriter_write = (
        getattr(ret, "__module__", None) + "." + ret.__class__.__name__
        if getattr(ret, "__module__", None)
        else ret.__class__.__name__
    )
    print(
        "✅ pandas.io.json._json.FrameWriter.write:",
        type_pandas_io_json__json_FrameWriter_write,
    )
except Exception as e:
    type_pandas_io_json__json_FrameWriter_write = "_syft_missing"
    print("❌ pandas.io.json._json.FrameWriter.write: Return unavailable")
    print("  Please fix this return type code until there is no exception")
    print("   Error:", e)
