#!/usr/bin/env python
# coding: utf-8

# ## sklearn.linear_model._logistic.LogisticRegression

# In[31]:


# third party
import numpy as np
import sklearn
import sklearn.linear_model

x = np.array([1, 2, 4, 5, 6]).reshape(-1, 1)
y = np.array([0, 0, 1, 1, 1])


def class_constructor(*args, **kwargs):
    obj = sklearn.linear_model._logistic.LogisticRegression().fit(x, y)
    return obj


# In[65]:


# sklearn.linear_model._logistic.LogisticRegression.__getstate__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__getstate__()
    state = ret
    type_sklearn_linear_model__logistic_LogisticRegression___getstate__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.__getstate__: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression___getstate__ = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.__getstate__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[33]:


# sklearn.linear_model._logistic.LogisticRegression.__repr__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__repr__()
    type_sklearn_linear_model__logistic_LogisticRegression___repr__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.__repr__: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression___repr__ = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.__repr__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[66]:


# sklearn.linear_model._logistic.LogisticRegression.__setstate__
try:
    obj = class_constructor()  # noqa F821
    ret = obj.__setstate__(state)
    type_sklearn_linear_model__logistic_LogisticRegression___setstate__ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.__setstate__: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression___setstate__ = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.__setstate__: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[62]:


# sklearn.linear_model._logistic.LogisticRegression._check_n_features
try:
    obj = class_constructor()  # noqa F821
    ret = obj._check_n_features(x, True)
    type_sklearn_linear_model__logistic_LogisticRegression__check_n_features = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._check_n_features: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__check_n_features = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._check_n_features: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[36]:


# sklearn.linear_model._logistic.LogisticRegression._get_param_names
try:
    obj = class_constructor()  # noqa F821
    ret = obj._get_param_names()
    type_sklearn_linear_model__logistic_LogisticRegression__get_param_names = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._get_param_names: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__get_param_names = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._get_param_names: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[37]:


# sklearn.linear_model._logistic.LogisticRegression._get_tags
try:
    obj = class_constructor()  # noqa F821
    ret = obj._get_tags()
    type_sklearn_linear_model__logistic_LogisticRegression__get_tags = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._get_tags: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__get_tags = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._get_tags: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[38]:


# sklearn.linear_model._logistic.LogisticRegression._more_tags
try:
    obj = class_constructor()  # noqa F821
    ret = obj._more_tags()
    type_sklearn_linear_model__logistic_LogisticRegression__more_tags = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._more_tags: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__more_tags = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._more_tags: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[61]:


# sklearn.linear_model._logistic.LogisticRegression._predict_proba_lr
try:
    obj = class_constructor()  # noqa F821
    ret = obj._predict_proba_lr(x)
    type_sklearn_linear_model__logistic_LogisticRegression__predict_proba_lr = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._predict_proba_lr: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__predict_proba_lr = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._predict_proba_lr: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[40]:


# sklearn.linear_model._logistic.LogisticRegression._repr_html_
try:
    obj = class_constructor()
    ret = obj._repr_html_
    type_sklearn_linear_model__logistic_LogisticRegression__repr_html_ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._repr_html_:", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__repr_html_ = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._repr_html_: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[41]:


# sklearn.linear_model._logistic.LogisticRegression._repr_html_inner
try:
    obj = class_constructor()  # noqa F821
    ret = obj._repr_html_inner()
    type_sklearn_linear_model__logistic_LogisticRegression__repr_html_inner = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._repr_html_inner: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__repr_html_inner = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._repr_html_inner: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[42]:


# sklearn.linear_model._logistic.LogisticRegression._repr_mimebundle_
try:
    obj = class_constructor()  # noqa F821
    ret = obj._repr_mimebundle_()
    type_sklearn_linear_model__logistic_LogisticRegression__repr_mimebundle_ = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._repr_mimebundle_: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__repr_mimebundle_ = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._repr_mimebundle_: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[60]:


# sklearn.linear_model._logistic.LogisticRegression._validate_data
try:
    obj = class_constructor()  # noqa F821
    ret = obj._validate_data(x)
    type_sklearn_linear_model__logistic_LogisticRegression__validate_data = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression._validate_data: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression__validate_data = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression._validate_data: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[59]:


# sklearn.linear_model._logistic.LogisticRegression.decision_function
try:
    obj = class_constructor()  # noqa F821
    ret = obj.decision_function(x)
    type_sklearn_linear_model__logistic_LogisticRegression_decision_function = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.decision_function: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_decision_function = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.decision_function: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[45]:


# sklearn.linear_model._logistic.LogisticRegression.densify
try:
    obj = class_constructor()  # noqa F821
    ret = obj.densify()
    type_sklearn_linear_model__logistic_LogisticRegression_densify = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.densify: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_densify = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.densify: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[58]:


# sklearn.linear_model._logistic.LogisticRegression.fit
try:
    obj = class_constructor()  # noqa F821
    ret = obj.fit(x, y)
    type_sklearn_linear_model__logistic_LogisticRegression_fit = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.fit: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_fit = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.fit: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[47]:


# sklearn.linear_model._logistic.LogisticRegression.get_params
try:
    obj = class_constructor()  # noqa F821
    ret = obj.get_params()
    type_sklearn_linear_model__logistic_LogisticRegression_get_params = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.get_params: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_get_params = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.get_params: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[57]:


# sklearn.linear_model._logistic.LogisticRegression.predict
try:
    obj = class_constructor()  # noqa F821
    ret = obj.predict(x)
    type_sklearn_linear_model__logistic_LogisticRegression_predict = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.predict: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_predict = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.predict: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[56]:


# sklearn.linear_model._logistic.LogisticRegression.predict_log_proba
try:
    obj = class_constructor()  # noqa F821
    ret = obj.predict_log_proba(x)
    type_sklearn_linear_model__logistic_LogisticRegression_predict_log_proba = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.predict_log_proba: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_predict_log_proba = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.predict_log_proba: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[55]:


# sklearn.linear_model._logistic.LogisticRegression.predict_proba
try:
    obj = class_constructor()  # noqa F821
    ret = obj.predict_proba(x)
    type_sklearn_linear_model__logistic_LogisticRegression_predict_proba = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.predict_proba: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_predict_proba = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.predict_proba: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[54]:


# sklearn.linear_model._logistic.LogisticRegression.score
try:
    obj = class_constructor()  # noqa F821
    ret = obj.score(x, y)
    type_sklearn_linear_model__logistic_LogisticRegression_score = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.score: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_score = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.score: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[52]:


# sklearn.linear_model._logistic.LogisticRegression.set_params
try:
    obj = class_constructor()  # noqa F821
    ret = obj.set_params()
    type_sklearn_linear_model__logistic_LogisticRegression_set_params = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.set_params: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_set_params = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.set_params: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[53]:


# sklearn.linear_model._logistic.LogisticRegression.sparsify
try:
    obj = class_constructor()  # noqa F821
    ret = obj.sparsify()
    type_sklearn_linear_model__logistic_LogisticRegression_sparsify = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.linear_model._logistic.LogisticRegression.sparsify: ", type(ret)
    )  # noqa E501
except Exception as e:
    type_sklearn_linear_model__logistic_LogisticRegression_sparsify = "_syft_missing"
    print(
        "❌ sklearn.linear_model._logistic.LogisticRegression.sparsify: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:
