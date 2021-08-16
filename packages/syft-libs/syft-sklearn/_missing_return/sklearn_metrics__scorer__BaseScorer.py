#!/usr/bin/env python
# coding: utf-8

# ## sklearn.metrics._scorer._BaseScorer

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.metrics._scorer._BaseScorer()
    return obj


# In[ ]:


# sklearn.metrics._scorer._BaseScorer.__call__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__call__()
    type_sklearn_metrics__scorer__BaseScorer___call__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.metrics._scorer._BaseScorer.__call__: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_metrics__scorer__BaseScorer___call__ = "_syft_missing"
    print(
        "❌ sklearn.metrics._scorer._BaseScorer.__call__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.metrics._scorer._BaseScorer.__repr__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__repr__()
    type_sklearn_metrics__scorer__BaseScorer___repr__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print("✅ sklearn.metrics._scorer._BaseScorer.__repr__: ", type(ret))  # noqa E501
except Exception as e:
    type_sklearn_metrics__scorer__BaseScorer___repr__ = "_syft_missing"
    print(
        "❌ sklearn.metrics._scorer._BaseScorer.__repr__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.metrics._scorer._BaseScorer._check_pos_label
try:
    obj = class_constructor()  # noqa F821
    ret = obj._check_pos_label()
    type_sklearn_metrics__scorer__BaseScorer__check_pos_label = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.metrics._scorer._BaseScorer._check_pos_label: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_metrics__scorer__BaseScorer__check_pos_label = "_syft_missing"
    print(
        "❌ sklearn.metrics._scorer._BaseScorer._check_pos_label: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.metrics._scorer._BaseScorer._factory_args
try:
    obj = class_constructor()  # noqa F821
    ret = obj._factory_args()
    type_sklearn_metrics__scorer__BaseScorer__factory_args = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.metrics._scorer._BaseScorer._factory_args: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_metrics__scorer__BaseScorer__factory_args = "_syft_missing"
    print(
        "❌ sklearn.metrics._scorer._BaseScorer._factory_args: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.metrics._scorer._BaseScorer._select_proba_binary
try:
    obj = class_constructor()  # noqa F821
    ret = obj._select_proba_binary()
    type_sklearn_metrics__scorer__BaseScorer__select_proba_binary = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.metrics._scorer._BaseScorer._select_proba_binary: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_metrics__scorer__BaseScorer__select_proba_binary = "_syft_missing"
    print(
        "❌ sklearn.metrics._scorer._BaseScorer._select_proba_binary: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
