import pytest_cases
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize

from brevitas.graph.quantize import layerwise_quantize
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.rotation_utils import RotationWeightParametrization


@pytest_cases.parametrize(
    'kwargs',
    [
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'name_blacklist': [],
            'key': '0',
            'expected': "<class 'brevitas.nn.quant_linear.QuantLinear'>"},
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'name_blacklist': ['0'],
            'key': '0',
            'expected': "<class 'torch.nn.modules.linear.Linear'>"},
        {
            'model': nn.Sequential(nn.Sequential(nn.Linear(2, 3))),
            'name_blacklist': ['0'],
            'key': '0.0',
            'expected': "<class 'torch.nn.modules.linear.Linear'>"},
        {
            'model': nn.Sequential(nn.Sequential(nn.Linear(2, 3))),
            'name_blacklist': ['0.0'],
            'key': '0.0',
            'expected': "<class 'torch.nn.modules.linear.Linear'>"},])
def test_layerwise_quantize_blacklist(kwargs):
    key = kwargs['key']
    exp = kwargs['expected']
    del kwargs['key']
    del kwargs['expected']
    qmodel = layerwise_quantize(**kwargs)
    checked = False
    found_names = []
    for n, m in qmodel.named_modules():
        found_names.append(n)
        if n == key:
            mt = str(type(m))
            assert mt == exp, f"Expect module {n} to be type: {exp}, found type {mt}"
            checked = True
    assert checked, f"Layer named {key} not found. Layer names are: {found_names}"


@pytest_cases.parametrize(
    'kwargs',
    [
        {
            'model': nn.Sequential(nn.Linear(2, 3)),
            'rot_mat': torch.tensor([[1., -1.], [1., 1.]]) / torch.sqrt(torch.tensor(2.)),
            'rot_func': lambda tensor,
                        rot_mat,
                        K: torch.matmul(tensor, rot_mat),
            'key': '0',
            'expected': "<class 'torch.nn.utils.parametrize.ParametrizedQuantLinear'>"},])
def test_layerwise_quantize_parametrized_modules(kwargs):
    key = kwargs['key']
    exp = kwargs['expected']
    rot_mat = kwargs['rot_mat']
    rot_func = kwargs['rot_func']
    del kwargs['key']
    del kwargs['expected']
    del kwargs['rot_mat']
    del kwargs['rot_func']

    model = kwargs["model"]
    module = recurse_getattr(model, key)
    # Register rotation parametrization to module
    parametrize.register_parametrization(
        module=module,
        tensor_name="weight",
        parametrization=RotationWeightParametrization(
            rot_mat=nn.Parameter(rot_mat),
            rot_func=rot_func,
            axis=1,
            K=None,
        ))
    qmodel = layerwise_quantize(**kwargs)
    checked = False
    found_names = []
    for n, m in qmodel.named_modules():
        found_names.append(n)
        if n == key:
            mt = str(type(m))
            assert mt == exp, f"Expect module {n} to be type: {exp}, found type {mt}"
            checked = True
    assert checked, f"Layer named {key} not found. Layer names are: {found_names}"
