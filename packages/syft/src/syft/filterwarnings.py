# stdlib
import warnings

# its not clear what is causing this warning spam but it must be a third party numerical
# library changing numpy's smallest subnormal or something?
# https://github.com/numpy/numpy/issues/20895
warnings.filterwarnings(
    "ignore",
    ".*The value of the smallest subnormal for.*",
    category=UserWarning,
)

# DeprecationWarning: 'urllib3.contrib.pyopenssl' module is deprecated and will be
# removed in a future release of urllib3 2.x.
# Read more in this issue: https://github.com/urllib3/urllib3/issues/2680
warnings.filterwarnings(
    "ignore",
    ".*urllib3.contrib.pyopenssl.*",
    category=DeprecationWarning,
)
