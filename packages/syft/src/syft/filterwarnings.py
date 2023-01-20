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
