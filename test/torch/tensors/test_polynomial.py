import syft as sy
import torch as th


def test_polynomial_tensor():

    x_poly = th.zeros(3).polynomial()
    y = th.rand(3)
    y_poly = y.poly_constant()
    z_poly = x_poly + y_poly
    out_poly = z_poly * z_poly

    x = th.rand(3)
    z = x + y
    pred = z * z

    factors = out_poly.factors()
    vectorized_input = out_poly.poly_vectorize_input_vector(x)
    approx_pred = (factors * vectorized_input).sum(1)

    assert th.isclose(pred, approx_pred.float()).all()
