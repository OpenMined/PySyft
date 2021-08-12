#!/usr/bin/env python
# coding: utf-8

# ## sklearn.metrics._plot.roc_curve.RocCurveDisplay

# In[ ]:


import sklearn
def class_constructor(*args, **kwargs):
    obj = sklearn.metrics._plot.roc_curve.RocCurveDisplay()
    return obj


# In[ ]:


# sklearn.metrics._plot.roc_curve.RocCurveDisplay.plot
try:
    obj = class_constructor() # noqa F821
    ret = obj.plot()
    type_sklearn_metrics__plot_roc_curve_RocCurveDisplay_plot = getattr(ret, '__module__', 'none') + '.' + ret.__class__.__name__
    print('✅ sklearn.metrics._plot.roc_curve.RocCurveDisplay.plot: ', type(ret)) # noqa E501
except Exception as e:
    type_sklearn_metrics__plot_roc_curve_RocCurveDisplay_plot = '_syft_missing'
    print('❌ sklearn.metrics._plot.roc_curve.RocCurveDisplay.plot: Return unavailable') # noqa E501
    print("  Please fix this return type code until there is no exception")
    print('  Error:', e)

