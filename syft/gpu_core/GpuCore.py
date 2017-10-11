"""This modules helps to perform mathematical tensor operations on the host's GPU.

It translates standard tensor operations (e.g. matrix-multiplication, vector-element-addition, activation-functions...)
to the GPU, which results in overall better performance.

Requirements:
    numpy
    PyOpenCL
"""
import pyopencl as cl
import numpy as np

from syft.gpu_core.GpuBuffer import GpuBuffer
from syft.gpu_core.Gpu import Gpu
from syft.gpu_core.GpuProgram import GpuProgram


class GpuCore:
    """Handles low-level interactions with GPU."""
    GPU_DATA_TYPE = np.float32

    def __init__(self):
        # getting platform and devices as Python objects/OpenCL objects
        self.platforms, self.gpus, self.cl_gpus = self.__find_compute_devices()

        # create contect with GPU to start enqueueing operations
        self.context, self.queue = self.__connect_to_gpu(self.cl_gpus, self.platforms)

        # operations to be queued up to GPU
        self.programs = []


    def __find_compute_devices(self):
        """Looks for GPUs to be able to connect to.

        Returns:
            gpus    Gpu[]       Python Gpu object with necessary data for computing
        """
        # getting working platform
        platforms = cl.get_platforms()

        # get GPU compute devices
        cl_gpus = platforms[0].get_devices(device_type=cl.device_type.GPU)

        # check if GPUs were found and if return them as a Object
        if cl_gpus is []:
            raise Exception("No GPUs found.")

        gpus = []
        for cl_gpu in cl_gpus:
            # create a GPU object and append to return array
            gpu = Gpu(cl_gpu)
            gpus.append(gpu)

        return platforms, gpus, cl_gpus


    def __connect_to_gpu(self, cl_gpus, platforms):
        """Seeks for GPUs and initializes the contact to be able to perform operations later.

        Parameters
            cl_gpus     cl.Device[]         found OpenCL GPUs
            platforms   cl.Platform[]       found OpenCL platforms

        Returns:
            context     Context             context created with the GPUs
        """
        context = cl.Context(cl_gpus, properties=[(cl.context_properties.PLATFORM, platforms[0])])
        queue = cl.CommandQueue(context)

        return context, queue


    def __check_array_dtypes(self, *args):
        """Checks if array data types can be used on GPU.

        Parameters:
            args        ndarray         arrays that get their data type checked
        """
        for arg in args:
            if arg.dtype != self.GPU_DATA_TYPE:
                raise Exception(f"\n\n\tndarrays must be of dtype={self.GPU_DATA_TYPE} to be able to run on GPU.")


    def __check_array_shapes(self, *args):
        """Checks if array shapes are the same for every array.

        Parameters:
            args        ndarray         arrays that get their shape checked
        """
        for arg in args:
            if arg.shape != arg[0].shape:
                raise Exception(f"\n\n\tndarrays must be of same size.")


    def vec_add(self, v1, v2):
        """Prepares vector addition on GPU.

        Parameters
            v1          ndarray             vector1 numpy array
            v2          ndarray             vector2 numpy array
        """
        # controlling input
        self.__check_array_dtypes(v1, v2)
        self.__check_array_shapes()

        # creating input buffers
        buf1 = GpuBuffer(self.context, v1)
        buf2 = GpuBuffer(self.context, v2)

        # creating output buffer
        empty_arr = np.empty_like(v1)
        out_buf = GpuBuffer(self.context, empty_arr, is_out_buffer=True)

        # creating GPU program
        #                                         work_dim, local_dim, kernel_params, kernel_scalar_dtypes
        prg = GpuProgram(self.context, "VEC_ADD", v1.shape, None, [buf1, buf2, out_buf, len(v1)], [None, None, None, np.uint32])

        self.programs.append(prg)

        # returning the out buffer object that the GPU results will be written to
        return out_buf


    def run(self):
        """Creates a runnable GPU program from previously defined operations."""
        for program in self.programs:

            # running program on current queue
            program.run(self.queue)

            # copying the results to the buffer
            cl.enqueue_copy(self.queue, program.out_buffer.result, program.out_buffer.cl_buffer)

        # make sure that programs get executed on GPU
        self.queue.finish()