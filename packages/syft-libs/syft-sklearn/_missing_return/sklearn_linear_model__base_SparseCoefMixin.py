#!/usr/bin/env python
# coding: utf-8

# ## sklearn.linear_model._base.SparseCoefMixin

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.linear_model._base.SparseCoefMixin()
    return obj


# In[ ]:


# sklearn.linear_model._base.SparseCoefMixin.densify
try:
    obj = class_constructor()  # noqa F821
    ret = obj.densify()
    type_sklearn_linear_model__base_SparseCoefMixin_densify = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._base.SparseCoefMixin.densify: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__base_SparseCoefMixin_densify = "_syft_missing"
    print(
        "❌ sklearn.linear_model._base.SparseCoefMixin.densify: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.linear_model._base.SparseCoefMixin.sparsify
try:
    obj = class_constructor()  # noqa F821
    ret = obj.sparsify()
    type_sklearn_linear_model__base_SparseCoefMixin_sparsify = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._base.SparseCoefMixin.sparsify: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__base_SparseCoefMixin_sparsify = "_syft_missing"
    print(
        "❌ sklearn.linear_model._base.SparseCoefMixin.sparsify: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
