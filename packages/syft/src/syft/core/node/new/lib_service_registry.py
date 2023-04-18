# third party
import numpy

api_registry_libs = [numpy]

function_signatures_registry = {
    "concatenate": "concatenate(a1,a2, *args,axis=0,out=None,dtype=None,casting='same_kind')",
    "set_numeric_ops": "set_numeric_ops(op1=func1,op2=func2, *args)",
    "geterrorobj": "geterrobj()",
    "source": "source(object, output)",
}
