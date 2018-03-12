"""
Import Tests

The purpose of these tests is to break if you change a public api.

If anything in here breaks, it means you probably moved something.
If something breaks, PLEASE update the notebooks.
They also depend on these locations.
"""

def test_import_grid():
    import grid
    assert True

def test_import_keras_client():
    import grid.clients.keras
    assert True

def test_import_torch_client():
    import grid.clients.torch
    assert True

def test_import_lib():
    import grid.lib
    assert True

def test_import_utils():
    import grid.lib.utils
    assert True

def test_import_workers():
    import grid.workers
    assert True

def test_import_anchor_node():
    import grid.workers.anchor
    assert True

def test_import_compute_node():
    import grid.workers.compute
    assert True

def test_import_tree_node():
    import grid.workers.tree
    assert True
