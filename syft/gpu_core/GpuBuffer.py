import pyopencl as cl
import numpy as np

class GpuBuffer:
    """Creates a buffer object in python that loads to the GPU."""
    def __init__(self, context, ndarray, is_out_buffer=False):
        """Defining necessary attributes and creates a cl.Buffer.

        Parameters:
            context         cl.Context      OpenCL context with GPU

        Arguments:
            ndarray         ndarray         array of data to be buffered to GPU
                                            if is_out_buffer=True, the ndarray works as buffer object for receiving GPU data
            is_out_buffer   Bool            determines if this buffer is an out buffer if yes
        """
        # assigning necessary data for gpu queueing and checking if ndarray exceeds GPU capabilities
        if ndarray is not None:
            self.dim = self.__check_dimension(ndarray)

        self.context = context

        self.type = ndarray.dtype
        self.byte_size = ndarray.nbytes

        if is_out_buffer:
            self.result = ndarray
            self.is_out_buffer = True

        else:
            self.ndarray = ndarray
            self.is_out_buffer = False

        self.cl_buffer = self.__create_buffer()


    def __check_dimension(self, ndarray):
        """Checks if dimension is < 3, since OpenCL only allows three axes of data.

        Parameters:
            ndarray         ndarray         array of data to be checked the dimensions for

        Return:
            dim     int     dimension of input array
        """
        if ndarray.ndim < 3:
            return ndarray.ndim
        else:
            raise Exception("Currently supported are only up to 3 dimensions per array.")


    def __create_buffer(self):
        """Creates a Buffer that can be used to allocate memory on GPU.

        Out buffers get a separate treatment, because they do not posses host data

        Return:
            cl_buffer       cl.Buffer           OpenCL Buffer Object with data
        """
        if self.is_out_buffer:
            return cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, size=self.byte_size)

        else:
            return cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.ndarray)


