import syft

if syft.dependency_check.tenseal_available:
    from tenseal import context, ckks_vector, SCHEME_TYPE
    from _tenseal_cpp import CKKSVector
