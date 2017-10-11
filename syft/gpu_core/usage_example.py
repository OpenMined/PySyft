"""This shows an example of how to use the GPU computation API"""
import numpy as np

# We need to import our GpuCore()
from syft.gpu_core.GpuCore import GpuCore

############################################################ LONG GUIDE ################################################################

# For my system GPU computation exceeds CPU computation at about 900.000 FLOPs so let's chose a data length of this size
LENGTH = 900000

# Let's create our numpy arrays. These represent our TensorBase().data
t1 = np.ones(LENGTH).astype(np.float32)
t2 = np.random.rand(LENGTH).astype(np.float32)

t3 = np.ones(LENGTH).astype(np.float32)
t4 = np.random.rand(LENGTH).astype(np.float32)


# First we create a GpuCore() object.  The API's final goal is to compute all tensor operations of
# e.g. an artificial neural net on the GPU in the background, without the user ever needing to call GpuCore().
core = GpuCore()


# Then we decide which operation we would like to outsource the GPU
# (Currently, there is only vector addition available, but that will change soon).
out1 = core.vec_add(t1, t2)

# out1 is a GpuBuffer() object which is marked as an out buffer and therefore has an attribute .result which will
# give us our result after running our core.


# We can enqueue multiple operations to the GPU. They will be processed one after another
out2 = core.vec_add(t3, t4)


# Let's create a benchmark test
from time import time


def benchmark(func, *args):
    time1 = time()

    out = func(*args)

    time2 = time()

    print("\n\tfunction ran for:", time2 - time1, "sec.")
    return out


# Let's compute our vector addition normally on CPU
def vec_add_on_cpu(v1, v2):
    result = np.empty(len(v1))

    for i in range(len(v1)):
        result[i] = v1[i] + v2[i]

    return result

# Now we run our core on GPU and our normal function on CPU
benchmark(core.run)
result = benchmark(vec_add_on_cpu, t1, t2)

print("\n\tGPU result:", out1.result)
print("\tCPU result:", result)

print("\n\tAnd we realize that we just ran two vector additions on our GPU.\n\tHere is the result of the second operation:", out2.result)


# Conclusion: At large FLOPs we outperform the CPU significantly!

############################################################ LONG GUIDE END ############################################################



############################################################ SWIFT GUIDE ###############################################################

# Create a GpuCore() object.
core = GpuCore()


# Enqueue all ndarray-operations you want to perform on the GPU (e.g. vector addition) and which are supported
# (see /kernels for all supported operations).
out = core.vec_add(t1, t2)
# ...
# ...
# ...
# out is a GpuBuffer() object that has an attribute .result which will contain the result that operation after you ran
# the program on the GPU


# Finally run the Operations on the GPU
core.run()


# retrieve results (ndarray) using .result on GpuBuffer() objects
result = out.result

############################################################ SWIFT GUIDE END ###########################################################
