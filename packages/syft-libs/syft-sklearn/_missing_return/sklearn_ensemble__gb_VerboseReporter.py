#!/usr/bin/env python
# coding: utf-8

# ## sklearn.ensemble._gb.VerboseReporter

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.ensemble._gb.VerboseReporter()
    return obj


# In[ ]:


# sklearn.ensemble._gb.VerboseReporter.init
try:
    obj = class_constructor()  # noqa F821
    ret = obj.init()
    type_sklearn_ensemble__gb_VerboseReporter_init = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.ensemble._gb.VerboseReporter.init: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_ensemble__gb_VerboseReporter_init = "_syft_missing"
    print(
        "❌ sklearn.ensemble._gb.VerboseReporter.init: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.ensemble._gb.VerboseReporter.update
try:
    obj = class_constructor()  # noqa F821
    ret = obj.update()
    type_sklearn_ensemble__gb_VerboseReporter_update = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.ensemble._gb.VerboseReporter.update: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_ensemble__gb_VerboseReporter_update = "_syft_missing"
    print(
        "❌ sklearn.ensemble._gb.VerboseReporter.update: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
