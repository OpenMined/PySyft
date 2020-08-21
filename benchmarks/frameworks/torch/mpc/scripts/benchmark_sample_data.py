# this is sample data for benchmark of sigmoid function

# data format ('method_name', precision value)
benchmark_data_sigmoid = [("chebyshev", 4), ("maclaurin", 4), ("exp", 4)]


############################################################
#        Benchmark Data Additive Sharing Tensors            #
###########################################################

# data format benchmark_share_get_plot: ('protocol', dtype, n_workers)
b_data_share_get = [("int", 2), ("long", 2), ("int", 3), ("long", 3)]


# data format for  benchmark_max_pool2d_plot: (list) (['protocols'])
b_data_max_pool2d = ["snn", "fss"]
