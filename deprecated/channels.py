# Channels

# Main channel that high level commands are broadcasted on
om = 'openmined'

list_tasks = f'{om}:list_tasks'
add_task = f'{om}:task:add'
list_models = f'{om}:list_models'
list_workers = f'{om}:list_workers'


def list_tasks_callback(id):
    return f'{list_tasks}:{id}'


def list_workers_callback(id):
    return f'{list_workers}:{id}'


def add_model(name):
    return f'{add_task}:{name}'


# Whoami Channels

whoami_listener = f'{om}:whoami'
def whoami_listener_callback(id):
    return f'{whoami_listener}:{id}'


# Torch Channels

torch_listen_for_obj = f'{om}:torch_listen_for_obj'
def torch_listen_for_obj_callback(id):
    return f'{torch_listen_for_obj}:{id}'


torch_listen_for_obj_response = f'{om}:torch_listen_for_obj_res'
def torch_listen_for_obj_response_callback(id):
    return f'{torch_listen_for_obj_response}:{id}'


torch_listen_for_obj_req = f'{om}:torch_listen_for_obj_req'
def torch_listen_for_obj_req_callback(id):
    return f'{torch_listen_for_obj_req}:{id}'


torch_listen_for_obj_req_response = f'{om}:torch_listen_for_obj_req_res'
def torch_listen_for_obj_req_response_callback(id):
    return f'{torch_listen_for_obj_req_response}:{id}'

torch_listen_for_command = f'{om}:torch_listen_for_command'
def torch_listen_for_command_callback(id):
    return f'{torch_listen_for_command}:{id}'

torch_listen_for_command_response = f'{om}:torch_listen_for_command_response'
def torch_listen_for_command_response_callback(id):
    return f'{torch_listen_for_command_response}:{id}'
