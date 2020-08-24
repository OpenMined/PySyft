"""
Sample data for the benchmarking of FPT/AST with differnt types of approximations
See: https://github.com/OpenMined/PySyft/issues/3997

data format: list[('method_name', precision value)]
"""

benchmark_data_sigmoid = [("chebyshev", 4), ("maclaurin", 4), ("exp", 4)]
benchmark_data_tanh = [("chebyshev", 4), ("sigmoid", 4)]

############################################################
#        Benchmark Data Additive Sharing Tensors          #
###########################################################

# data format benchmark_share_get_plot: ('protocol', dtype, n_workers)
b_data_share_get = [("int", 2), ("long", 2), ("int", 3), ("long", 3)]

# data format for  benchmark_max_pool2d_plot: (list) (['protocols'])
b_data_max_pool2d = ["snn", "fss"]

# data format for  benchmark_avg_pool2d_plot: (list) (['protocols'])
b_data_avg_pool2d = ["snn", "fss"]

# data format for  benchmark_avg_pool2d_plot: (list) (['protocols'])
b_data_batch_norm = ["snn", "fss"]
