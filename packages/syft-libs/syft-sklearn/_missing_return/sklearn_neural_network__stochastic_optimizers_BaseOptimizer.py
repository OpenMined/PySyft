#!/usr/bin/env python
# coding: utf-8

# ## sklearn.neural_network._stochastic_optimizers.BaseOptimizer

# In[ ]:


# third party
import sklearn


def class_constructor(*args, **kwargs):
    obj = sklearn.neural_network._stochastic_optimizers.BaseOptimizer()
    return obj


# In[ ]:


# sklearn.neural_network._stochastic_optimizers.BaseOptimizer.iteration_ends
try:
    obj = class_constructor()  # noqa F821
    ret = obj.iteration_ends()
    type_sklearn_neural_network__stochastic_optimizers_BaseOptimizer_iteration_ends = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.neural_network._stochastic_optimizers.BaseOptimizer.iteration_ends: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_neural_network__stochastic_optimizers_BaseOptimizer_iteration_ends = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.neural_network._stochastic_optimizers.BaseOptimizer.iteration_ends: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.neural_network._stochastic_optimizers.BaseOptimizer.trigger_stopping
try:
    obj = class_constructor()  # noqa F821
    ret = obj.trigger_stopping()
    type_sklearn_neural_network__stochastic_optimizers_BaseOptimizer_trigger_stopping = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.neural_network._stochastic_optimizers.BaseOptimizer.trigger_stopping: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_neural_network__stochastic_optimizers_BaseOptimizer_trigger_stopping = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.neural_network._stochastic_optimizers.BaseOptimizer.trigger_stopping: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)


# In[ ]:


# sklearn.neural_network._stochastic_optimizers.BaseOptimizer.update_params
try:
    obj = class_constructor()  # noqa F821
    ret = obj.update_params()
    type_sklearn_neural_network__stochastic_optimizers_BaseOptimizer_update_params = (
        getattr(ret, "__module__", "none") + "." + ret.__class__.__name__
    )
    print(
        "✅ sklearn.neural_network._stochastic_optimizers.BaseOptimizer.update_params: ",
        type(ret),
    )  # noqa E501
except Exception as e:
    type_sklearn_neural_network__stochastic_optimizers_BaseOptimizer_update_params = (
        "_syft_missing"
    )
    print(
        "❌ sklearn.neural_network._stochastic_optimizers.BaseOptimizer.update_params: Return unavailable"
    )  # noqa E501
    print("  Please fix this return type code until there is no exception")
    print("  Error:", e)
