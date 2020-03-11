import torch

from syft.generic import object_storage


def test_clear_objects():
    """ 
    
    Asserts that objects in storage are cleared 
    
    """
    obj_storage = object_storage.ObjectStorage() #obj_storage is a wrapper object to a collection of objects

    x = torch.tensor(1)
    obj_storage.set_obj(x)

    objs = obj_storage.current_objects() #Returns a copy of the objects present in obj_storage (here : x)

    assert len(objs) == 1
    assert objs[x.id] == x

    ret_val = obj_storage.clear_objects() #Completely removes all objects from obj_storage

    objs = obj_storage.current_objects()
    assert len(objs) == 0
    assert ret_val == obj_storage


def test_clear_objects_return_None():
    """ 
   
    Asserts that objects are cleared and returned as none 
   
    """
    obj_storage = object_storage.ObjectStorage()

    x = torch.tensor(1)
    obj_storage.set_obj(x)

    objs = obj_storage.current_objects()

    assert len(objs) == 1
    assert objs[x.id] == x

    ret_val = obj_storage.clear_objects(return_self=False)

    objs = obj_storage.current_objects()
    assert len(objs) == 0
    assert ret_val is None
