import pyopencl as cl

class Gpu:
    """Represents a GPU as a object with necessary computational info"""
    def __init__(self, gpu):
        """Assigns all necessary attributes that are needed to be known to effectivly compute on the GPU.

        Parameters:
            gpu         cl.Device       through OpenCL found GPU
        """
        # get the GPU vendor (e.g. NVIDIA, AMD...)
        self.VENDOR = gpu.get_info(cl.device_info.VENDOR)

        # get necessary computational info on GPU
        self.MAX_COMPUTE_UNITS = gpu.get_info(cl.device_info.MAX_COMPUTE_UNITS)
        self.MAX_WORK_GROUP_SIZE = gpu.get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self.MAX_WORK_ITEM_DIMENSIONS = gpu.get_info(cl.device_info.MAX_WORK_ITEM_DIMENSIONS)
        self.MAX_WORK_ITEM_SIZES = gpu.get_info(cl.device_info.MAX_WORK_ITEM_SIZES)

        self.LOCAL_MEM_SIZE = gpu.get_info(cl.device_info.LOCAL_MEM_SIZE)
        self.GLOBAL_MEM_SIZE = gpu.get_info(cl.device_info.GLOBAL_MEM_SIZE)

        # get the endian type
        self.ENDIAN_LITTLE = bool(gpu.get_info(cl.device_info.ENDIAN_LITTLE))

        # check if IEEE-754 64-bit floating point operations are supported
        self.IEE_754 = gpu.get_info(cl.device_info.DOUBLE_FP_CONFIG) is cl.device_fp_config.FMA

        # checks if device is available
        self.AVAILABLE = bool(gpu.get_info(cl.device_info.AVAILABLE))


    def __str__(self):
        return f"""
--------------------- GPU ---------------------
    Vendor:
        Vendor: {self.VENDOR}

    Computational Capabilities:
        max compute-units: {self.MAX_COMPUTE_UNITS}
        max work-group size: {self.MAX_WORK_GROUP_SIZE}
        max work-item dims: {self.MAX_WORK_ITEM_DIMENSIONS}
        max work-item sizes: {self.MAX_WORK_ITEM_SIZES}

    Memory Info:
        local-memory size: {self.LOCAL_MEM_SIZE}
        global-memory size: {self.GLOBAL_MEM_SIZE}

    Endian Type:
        little endian: {self.ENDIAN_LITTLE}

    Floating Point Configuration:
        IEE754-2008 support: {self.IEE_754}
        
    Availability:
        available: {self.AVAILABLE}
--------------------- END ---------------------
"""



