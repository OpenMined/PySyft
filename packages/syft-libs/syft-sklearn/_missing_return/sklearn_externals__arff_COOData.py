#!/usr/bin/env python
# coding: utf-8

# ## sklearn.externals._arff.COOData

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.externals._arff.COOData()
    return obj


# In[ ]:


# sklearn.externals._arff.COOData.decode_rows
try:
    obj = class_constructor()  # noqa F821
    ret = obj.decode_rows()
    type_sklearn_externals__arff_COOData_decode_rows = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.externals._arff.COOData.decode_rows: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_externals__arff_COOData_decode_rows = "_syft_missing"
    print(
        "❌ sklearn.externals._arff.COOData.decode_rows: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.externals._arff.COOData.encode_data
try:
    obj = class_constructor()  # noqa F821
    ret = obj.encode_data()
    type_sklearn_externals__arff_COOData_encode_data = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.externals._arff.COOData.encode_data: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_externals__arff_COOData_encode_data = "_syft_missing"
    print(
        "❌ sklearn.externals._arff.COOData.encode_data: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
