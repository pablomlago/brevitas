# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
from functools import partial
import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.gpxq import gpxq_stats_wrap
from brevitas.graph.qronos import Qronos
from brevitas.graph.utils import get_module_name_and_parent
from brevitas.graph.utils import gpxq_compute_error_stats
from brevitas.graph.utils import set_module
import brevitas.nn as qnn
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from brevitas.utils.stats_utils import collect_stats
from brevitas.utils.stats_utils import DictStatsCollector
from tests.conftest import SEED

IN_FEATURES = 2
OUT_FEATURES = 2

torch.manual_seed(SEED)


class SubModel(nn.Module):

    def __init__(self):
        super(SubModel, self).__init__()
        self.linear = nn.Linear(IN_FEATURES, OUT_FEATURES)

    def forward(self, x):
        return self.linear(x)


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        self.sub_model = SubModel()

    def forward(self, x):
        return self.sub_model(x)


def test_get_module_name_and_parent():
    model = Model()
    module_name, supermodule = get_module_name_and_parent(model, "sub_model.linear")
    assert module_name == "linear"
    assert supermodule is model.sub_model


def test_set_module():
    model = Model()
    new_module = nn.Linear(IN_FEATURES, OUT_FEATURES)
    set_module(model, new_module, "sub_model.linear")
    assert model.sub_model.linear is new_module


# ---------------------------------------------------------------------------
# Tests for gpxq_compute_error_stats
# ---------------------------------------------------------------------------

GPXQ_IN_FEATURES = 4
GPXQ_OUT_FEATURES = 8
GPXQ_SEQLEN = 16
GPXQ_SEED = 42
GPXQ_NUM_SAMPLES = 8

# Map optimizer name to (class, context-manager constructor)
_OPT_REGISTRY = {
    'gptq': (GPTQ, gptq_mode),
    'qronos': (Qronos, partial(gpfq_mode, algorithm_impl=Qronos)),}


class QuantLinearModel(nn.Module):
    """Two-layer model for GPxQ stats tests."""

    def __init__(self):
        super().__init__()
        self.linear_1 = qnn.QuantLinear(
            GPXQ_IN_FEATURES,
            GPXQ_OUT_FEATURES,
            bias=False,
            weight_quant=Int8WeightPerTensorFloat.let(
                bit_width=3, scaling_impl_type="parameter_from_stats"))
        self.linear_2 = qnn.QuantLinear(
            GPXQ_OUT_FEATURES,
            GPXQ_IN_FEATURES,
            bias=False,
            weight_quant=Int8WeightPerTensorFloat.let(
                bit_width=3, scaling_impl_type="parameter_from_stats"))

    def forward(self, x):
        return self.linear_2(self.linear_1(x))


def _build_model_and_calibration_data():
    """Create a :class:`QuantLinearModel` with fixed weights and deterministic
    calibration data of shape ``[GPXQ_NUM_SAMPLES, GPXQ_SEQLEN, GPXQ_IN_FEATURES]``."""
    torch.manual_seed(GPXQ_SEED)
    model = QuantLinearModel()
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, GPXQ_SEQLEN, GPXQ_IN_FEATURES))
    torch.manual_seed(GPXQ_SEED + 1)
    calib_input = torch.randn(GPXQ_NUM_SAMPLES, GPXQ_SEQLEN, GPXQ_IN_FEATURES)
    return model, calib_input


def _get_layer_inputs(model, calib_input, layer):
    """Return ``(X_quant, X_float)`` — the activations produced by *layer*
    with quantised vs float weights.

    These correspond to the inputs that the *next* layer would see during
    the two forward passes of ``gpfq_mode.catch_stopfwd``."""
    with torch.no_grad():
        X_quant = layer(calib_input)
        with quantization_status_manager(model,
                                         disable_act_quant=True,
                                         disable_weight_quant=True,
                                         disable_bias_quant=True,
                                         is_training=False):
            X_float = layer(calib_input)
    return X_quant, X_float


def _run_gpxq_with_stats(model, calib_input, opt='gptq', captured_matrices=None):
    """Run a GPxQ pass with a no-op ``_single_layer_update`` and return the
    collected :class:`DictStatsCollector`.

    If *captured_matrices* is a dict, raw ``H``, ``G``, ``R`` tensors are
    stored under ``"{layer}_{prefix}_{matrix}"`` keys."""

    @gpxq_stats_wrap
    def _noop(self):
        pass

    opt_class, opt_mode = _OPT_REGISTRY[opt]

    collector = DictStatsCollector()
    with collect_stats(collector):
        with patch.object(opt_class, '_single_layer_update', _noop):
            with torch.no_grad():
                with opt_mode(model, use_quant_activations=False, create_weight_orig=True) as ctx:
                    if captured_matrices is not None:

                        def _capture(prefix, name, layer, H, G=None, R=None, **kw):
                            captured_matrices[f"{name}_{prefix}_H"] = H.clone()
                            if G is not None:
                                captured_matrices[f"{name}_{prefix}_G"] = G.clone()
                            if R is not None:
                                captured_matrices[f"{name}_{prefix}_R"] = R.clone()
                            return gpxq_compute_error_stats(
                                prefix=prefix, name=name, layer=layer, H=H, G=G, R=R)

                        collector.on("pre_update", functools.partial(_capture, prefix="pre"))
                        collector.on("post_update", functools.partial(_capture, prefix="post"))

                    gpxq_model = ctx.model
                    for _ in range(ctx.num_layers):
                        gpxq_model(calib_input)
                        ctx.update()
    return collector


def _compute_expected_covariance(left, right=None, qronos=False):
    """Compute the expected normalised covariance ``left^T @ right / T`` the
    same way ``process_input`` + ``update_batch`` would.

    Parameters
    ----------
    left, right : Tensor of shape ``[N, S, C]``
        If *right* is ``None`` it defaults to *left* (self-correlation).
    qronos : bool
        Use Qronos normalisation (``1/T``) vs GPTQ (``2/T``).

    Returns
    -------
    Tensor of shape ``[1, C, C]``.
    """
    if right is None:
        right = left
    scale = (lambda T: math.sqrt(1.0 / T)) if qronos else (lambda T: math.sqrt(2.0 / T))

    def _prep(x):
        x = x.reshape(-1, x.shape[-1]).to(torch.float32)
        T = x.shape[0]
        return (scale(T) * x.t()).unsqueeze(0)  # [1, C, T]

    return _prep(left).bmm(_prep(right).transpose(2, 1))  # [1, C, C]


class TestGPXQStats:
    """Tests for the statistics collection machinery around GPxQ algorithms."""

    # -- Hessian / covariance capture tests --------------------------------

    def test_gptq_hessian_captured(self):
        """GPTQ Hessian forwarded to the stats callback must match the
        covariance accumulated during calibration."""
        model, calib_input = _build_model_and_calibration_data()

        captured = {}
        _run_gpxq_with_stats(model, calib_input, captured_matrices=captured)

        H_expected = _compute_expected_covariance(calib_input)
        for prefix in ("pre", "post"):
            torch.testing.assert_close(captured[f"linear_1_{prefix}_H"], H_expected)

    def test_qronos_matrices_captured(self):
        """Qronos H, G, R matrices forwarded to the stats callback must match
        hand-computed covariances for both layers."""
        model, calib_input = _build_model_and_calibration_data()
        X_quant_1, X_float_1 = _get_layer_inputs(model, calib_input, model.linear_1)

        captured = {}
        _run_gpxq_with_stats(model, calib_input, opt='qronos', captured_matrices=captured)

        # Expected covariances per layer
        expected = {
            "linear_1": {
                "H": _compute_expected_covariance(calib_input, qronos=True),
                # For layer 1, X_quant == X_float == calib_input, so G == H
                "G": _compute_expected_covariance(calib_input, qronos=True),},
            "linear_2": {
                "H": _compute_expected_covariance(X_quant_1, qronos=True),
                "G": _compute_expected_covariance(X_float_1, X_quant_1, qronos=True),
                "R": _compute_expected_covariance(X_float_1, qronos=True),},}

        for layer_name, matrices in expected.items():
            for mat_name, mat_expected in matrices.items():
                for prefix in ("pre", "post"):
                    key = f"{layer_name}_{prefix}_{mat_name}"
                    torch.testing.assert_close(captured[key], mat_expected, msg=key)

    # -- Error statistics tests --------------------------------------------

    @staticmethod
    def _assert_error_stats(stats, expected, prefix):
        """Assert that every key in *expected* matches the collected stats for
        the given *prefix* (``"pre"`` or ``"post"``)."""
        for key, value in expected.items():
            stat_key = f"{prefix}_{key}"
            assert stat_key in stats, f"Missing stat '{stat_key}'"
            assert stats[stat_key] == pytest.approx(value, abs=1e-5), stat_key

    def test_gptq_error_stats(self):
        """GPTQ error statistics must match values computed directly from the
        inputs and weights.

        Cross-checks the Hessian-domain identity
        ``||XE^T||_F^2 = tr(E H E^T)`` against the direct Frobenius-norm
        computation.  Since ``_single_layer_update`` is a no-op, pre- and
        post-update statistics are identical."""
        model, calib_input = _build_model_and_calibration_data()
        collector = _run_gpxq_with_stats(model, calib_input)

        layer = model.linear_1
        layer_name = "linear_1"
        assert layer_name in collector.stats
        stats = collector.stats[layer_name]

        dtype = torch.float32
        Q = layer.quant_weight().value.to(dtype=dtype)
        W = layer.weight_orig.to(dtype=dtype)
        err = Q - W
        X = calib_input.reshape(-1, GPXQ_IN_FEATURES).to(dtype)

        expected = {
            "rel_weight_err": (torch.norm(err, p='fro') / torch.norm(W, p='fro')).item(),
            "rel_out_err": (torch.norm(X @ err.T, p='fro') / torch.norm(X @ W.T, p='fro')).item(),}

        for prefix in ("pre", "post"):
            self._assert_error_stats(stats, expected, prefix)

    def test_qronos_error_stats(self):
        """Qronos error statistics — including the mismatched-objective
        ``fp_rel_out_err`` — must match direct computation.

        The mismatched objective is ``||X W^T - X_tilde Q^T||_F / ||X W^T||_F``
        where ``X`` / ``X_tilde`` are the float / quantised-weight inputs.
        Since ``_single_layer_update`` is a no-op, pre == post."""
        model, calib_input = _build_model_and_calibration_data()
        collector = _run_gpxq_with_stats(model, calib_input, opt='qronos')

        layer = model.linear_2
        layer_name = "linear_2"
        assert layer_name in collector.stats
        stats = collector.stats[layer_name]

        dtype = calib_input.dtype
        Q = layer.quant_weight().value.to(dtype=dtype)
        W = layer.weight_orig.to(dtype=dtype)
        err = Q - W

        X_quant, X_float = _get_layer_inputs(model, calib_input, model.linear_1)
        X_quant = X_quant.reshape(-1, GPXQ_OUT_FEATURES).to(dtype)
        X_float = X_float.reshape(-1, GPXQ_OUT_FEATURES).to(dtype)

        expected = {
            "rel_weight_err": (torch.norm(err, p='fro') / torch.norm(W, p='fro')).item(),
            "rel_out_err":
                (torch.norm(X_quant @ err.T, p='fro') / torch.norm(X_quant @ W.T, p='fro')).item(),
            "fp_rel_out_err": (
                torch.norm(X_float @ W.T - X_quant @ Q.T, p='fro') /
                torch.norm(X_float @ W.T, p='fro')).item(),}

        for prefix in ("pre", "post"):
            self._assert_error_stats(stats, expected, prefix)
