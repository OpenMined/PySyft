from syft.service.user.user_roles import ServiceRole

def compute_price(execution_time, role, method, service_path): 
    base_price_per_byte = 0.0000001
    price_per_byte = {
        ServiceRole.GUEST: base_price_per_byte,
        ServiceRole.DATA_SCIENTIST: 10*base_price_per_byte,
        ServiceRole.DATA_OWNER: base_price_per_byte,
        ServiceRole.ADMIN: base_price_per_byte
    }

    # we don't here, but one could use `service_path` to look-up precomputed number of bytes occupied by the given method in memory 
    service_method_bytecode_size = 100 

    # we don't here, but one could use `method` to dynamically compute the number of bytes occupied by user-defined functions in memory, 
    # e.g., by recursively traversing referenced objects in `method` and applying `sys.getsizeof(obj)` to each object 
    user_bytecode_size = 100 

    memory_price = (service_method_bytecode_size + user_bytecode_size) * price_per_byte[role]

    base_price_per_second = 0.001
    price_per_second = {
        ServiceRole.GUEST: base_price_per_second,
        ServiceRole.DATA_SCIENTIST: 10*base_price_per_second,
        ServiceRole.DATA_OWNER: base_price_per_second,
        ServiceRole.ADMIN: base_price_per_second
    }

    # `execution_time` is time taken for the function to execute, which can be indirectly related to CPU cycles
    cpu_price = execution_time * price_per_second[role]

    total_price = memory_price + cpu_price

    return total_price

