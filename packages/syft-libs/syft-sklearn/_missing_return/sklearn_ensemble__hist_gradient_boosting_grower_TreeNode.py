#!/usr/bin/env python
# coding: utf-8

# ## sklearn.ensemble._hist_gradient_boosting.grower.TreeNode

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.ensemble._hist_gradient_boosting.grower.TreeNode()
    return obj


# In[ ]:


# sklearn.ensemble._hist_gradient_boosting.grower.TreeNode.__lt__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__lt__()
    type_sklearn_ensemble__hist_gradient_boosting_grower_TreeNode___lt__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.ensemble._hist_gradient_boosting.grower.TreeNode.__lt__: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_ensemble__hist_gradient_boosting_grower_TreeNode___lt__ = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.ensemble._hist_gradient_boosting.grower.TreeNode.__lt__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.ensemble._hist_gradient_boosting.grower.TreeNode.set_children_bounds
try:
    obj = class_constructor()  # noqa F821
    ret = obj.set_children_bounds()
    type_sklearn_ensemble__hist_gradient_boosting_grower_TreeNode_set_children_bounds = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.ensemble._hist_gradient_boosting.grower.TreeNode.set_children_bounds: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_ensemble__hist_gradient_boosting_grower_TreeNode_set_children_bounds = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.ensemble._hist_gradient_boosting.grower.TreeNode.set_children_bounds: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
