"""
Sample data for the benchmarking of FPT/AST with differnt types of approximations
See: https://github.com/OpenMined/PySyft/issues/3997

data format: ('method_name', precision value)
"""

benchmark_data_sigmoid = [("chebyshev", 4), ("maclaurin", 4), ("exp", 4)]
benchmark_data_tanh = [("chebyshev", 4), ("sigmoid", 4)]
