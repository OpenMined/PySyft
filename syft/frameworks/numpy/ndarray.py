import numpy

class ndarray(numpy.ndarray):

  """
  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    args = []
    in_no = []
    for i, input_ in enumerate(inputs):
      if isinstance(input_, ndarray):
        in_no.append(i)
        args.append(input_.view(numpy.ndarray))
      else:
        args.append(input_)

    outputs = kwargs.pop('out', None)
    out_no = []
    if outputs:
      out_args = []
      for j, output in enumerate(outputs):
        if isinstance(output, ndarray):
          out_no.append(j)
          out_args.append(output.view(numpy.ndarray))
        else:
          out_args.append(output)
      kwargs['out'] = tuple(out_args)
    else:
      outputs = (None,) * ufunc.nout

    results = super(ndarray, self).__array_ufunc__(ufunc, method, *args, **kwargs)
    if results is NotImplemented:
      return NotImplemented

    results = tuple((numpy.asarray(result).view(ndarray)
                     if output is None else output)
                    for result, output in zip(results, outputs))


    return results[0] if len(results) == 1 else results
  """