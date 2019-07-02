import torch
import syft

# from syft.frameworks.torch.tensors.interpreters import syft.CRTTesor
# from syft import syft.CRTTesor


def test_wrap():
    """
    Test the .on() wrap functionality for syft.CRTTesor
    """

    res_3 = torch.tensor([[1, 2], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    res_7 = torch.tensor([[3, 4], [5, 6]]).fix_prec(field=7, precision_fractional=0)
    residues = {3: res_3, 7: res_7}

    x = syft.CRTTensor().on(residues, wrap=False).wrap()  # TODO see why I have to do that

    assert isinstance(x, torch.Tensor)
    assert isinstance(x.child, syft.CRTTensor)
    assert isinstance(x.child.child, dict)


def test__str__():
    res_3 = torch.tensor([[1, 2], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    res_7 = torch.tensor([[3, 4], [5, 6]]).fix_prec(field=7, precision_fractional=0)
    residues = {3: res_3, 7: res_7}

    crt = syft.CRTTensor(residues).wrap()

    assert isinstance(crt.__str__(), str)


def test_eq():
    res_3a = torch.tensor([1, 2]).fix_prec(field=3, precision_fractional=0)
    res_7a = torch.tensor([3, 4]).fix_prec(field=7, precision_fractional=0)
    residues_a = {3: res_3a, 7: res_7a}
    crt_a = syft.CRTTensor(residues_a).wrap()

    crt_b = syft.CRTTensor(residues_a).wrap()

    res_3c = torch.tensor([1, 2]).fix_prec(field=3, precision_fractional=0)
    res_7c = torch.tensor([4, 4]).fix_prec(field=7, precision_fractional=0)
    residues_c = {3: res_3c, 7: res_7c}
    crt_c = syft.CRTTensor(residues_c).wrap()
    
    eq_ab = (crt_a == crt_b).child.solve_system()
    eq_ac = (crt_a == crt_c).child.solve_system()

    exp_ab = torch.tensor([1, 1]).fix_prec(field=21, precision_fractional=0)
    exp_ac = torch.tensor([0, 1]).fix_prec(field=21, precision_fractional=0)
    
    assert ((eq_ab == exp_ab).all())
    assert ((eq_ac == exp_ac).all())


def test_add():
    res_3 = torch.tensor([[1, 2], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    res_7 = torch.tensor([[3, 4], [5, 6]]).fix_prec(field=7, precision_fractional=0)
    residues = {3: res_3, 7: res_7}

    crt1 = syft.CRTTensor(residues).wrap()
    crt2 = syft.CRTTensor(residues).wrap()
    
    result = crt1 + crt2
    
    exp_3 = torch.tensor([[2, 1], [0, 2]]).fix_prec(field=3, precision_fractional=0)
    exp_7 = torch.tensor([[6, 1], [3, 5]]).fix_prec(field=7, precision_fractional=0)
    exp_res = {3: exp_3, 7: exp_7}
    exp = syft.CRTTensor(exp_res).wrap()

    assert (result.child.solve_system() == exp.child.solve_system()).all()


def test_sub():
    res_a3 = torch.tensor([[1, 2], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    res_a7 = torch.tensor([[3, 4], [5, 6]]).fix_prec(field=7, precision_fractional=0)
    residues_a = {3: res_a3, 7: res_a7}

    res_b3 = torch.tensor([[1, 1], [0, 2]]).fix_prec(field=3, precision_fractional=0)
    res_b7 = torch.tensor([[5, 1], [5, 2]]).fix_prec(field=7, precision_fractional=0)
    residues_b = {3: res_b3, 7: res_b7}

    crt_a = syft.CRTTensor(residues_a).wrap()
    crt_b = syft.CRTTensor(residues_b).wrap()
    
    result = crt_a - crt_b
    
    exp_3 = torch.tensor([[0, 1], [0, 2]]).fix_prec(field=3, precision_fractional=0)
    exp_7 = torch.tensor([[5, 3], [0, 4]]).fix_prec(field=7, precision_fractional=0)
    exp_res = {3: exp_3, 7: exp_7}
    exp = syft.CRTTensor(exp_res).wrap()

    assert (result.child.solve_system() == exp.child.solve_system()).all()


def test_mul():
    res_3 = torch.tensor([[1, 2], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    res_7 = torch.tensor([[3, 4], [5, 6]]).fix_prec(field=7, precision_fractional=0)
    residues = {3: res_3, 7: res_7}

    crt1 = syft.CRTTensor(residues).wrap()
    crt2 = syft.CRTTensor(residues).wrap()
    
    result = crt1 * crt2
    
    exp_3 = torch.tensor([[1, 1], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    exp_7 = torch.tensor([[2, 2], [4, 1]]).fix_prec(field=7, precision_fractional=0)
    exp_res = {3: exp_3, 7: exp_7}
    exp = syft.CRTTensor(exp_res).wrap()

    assert (result.child.solve_system() == exp.child.solve_system()).all()


def test_torch_sum():
    res_3 = torch.tensor([[1, 2], [0, 1]]).fix_prec(field=3, precision_fractional=0)
    res_7 = torch.tensor([[3, 4], [5, 6]]).fix_prec(field=7, precision_fractional=0)
    residues = {3: res_3, 7: res_7}

    crt = syft.CRTTensor(residues)

    res = torch.sum(crt)
    assert res.child == {3: 1, 7:4}
