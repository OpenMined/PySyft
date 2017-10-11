import pyopencl as cl
from syft.gpu_core.GpuBuffer import GpuBuffer

class GpuProgram:
    """Represents a GPU program as an object."""
    def __init__(self, context,  kernel, global_work_size, local_work_size, params, scalar_arg_dtypes):
        """Defines attributes and creates a cl.Program.

        Parameters:
            context             cl.Context      OpenCL context with GPU
            kernel_str          String          kernel script that gets loaded
            global_work_size    (Int, Int, Int) defines the programs global working size on GPU in bytes
            local_work_size     (Int, Int, Int) defines the programs work group size on GPU in bytes
                                None            None indicates that work group size will be auto assigned by GPU
        """
        self.context = context

        self.scalar_arg_types = scalar_arg_dtypes

        # getting kernel script
        with open(f"syft/gpu_core/kernels/{kernel}.cl") as kernel_file:
            self.script = kernel_file.read()

        self.global_work_size = global_work_size
        self.local_work_size = local_work_size

        self.cl_program = self.__build_program(context)

        # getting constructing the input params for kernel function
        self.cl_params = []

        for param in params:
            if isinstance(param, GpuBuffer):
                self.cl_params.append(param.cl_buffer)

                # extract out_buffer
                if param.is_out_buffer:
                    self.out_buffer = param
            else:
                self.cl_params.append(param)


    def __build_program(self, context):
        """Creates a GPU runnable program from given inputs.

        Parameters:
            context     cl.Context      OpenCL context with GPU

        Returns:
            cl_program  cl.Program      OpenCL program runnable on GPU
        """
        # create program
        cl_program = cl.Program(context, self.script).build()

        return cl_program


    def run(self, queue):
        """Runs the program on GPU.

        Parameters:
            queue       cl.CommandQueue         queue to enqueue the program with
        """

        # retrieving program's kernel
        kernel = self.cl_program.all_kernels()[0]

        # setting argument types
        kernel.set_scalar_arg_dtypes(self.scalar_arg_types)

        # executing
        kernel(queue, self.global_work_size, self.local_work_size, *self.cl_params)




