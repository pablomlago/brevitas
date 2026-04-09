# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import functools
import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.gpxq import gpxq_stats_wrap
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


# TODO: Group under a class, e.g. TestGPXQStats
# ---------------------------------------------------------------------------
# Tests for gpxq_compute_error_stats
# ---------------------------------------------------------------------------

GPXQ_IN_FEATURES = 4
GPXQ_OUT_FEATURES = 8
GPXQ_SEQLEN = 16
GPXQ_SEED = 42
GPXQ_NUM_SAMPLES = 8


class QuantLinearModel(nn.Module):
    """Minimal single-layer model for GPTQ stats tests."""

    def __init__(self):
        super().__init__()
        self.linear = qnn.QuantLinear(
            GPXQ_IN_FEATURES,
            GPXQ_OUT_FEATURES,
            bias=False,
            weight_quant=Int8WeightPerTensorFloat.let(
                bit_width=3, scaling_impl_type="parameter_from_stats"))

    def forward(self, x):
        return self.linear(x)


def _build_model_and_calibration_data():
    """Create a QuantLinearModel with fixed weights and deterministic calibration data.

    Returns the model (eval mode, quantizer initialized) and the calibration
    input tensor of shape ``[GPXQ_NUM_SAMPLES, GPXQ_SEQLEN, GPXQ_IN_FEATURES]``.
    """
    torch.manual_seed(GPXQ_SEED)
    model = QuantLinearModel()
    model.eval()
    # Run a forward pass to initialise the weight quantizer (scale / zero-point)
    with torch.no_grad():
        model(torch.randn(1, GPXQ_SEQLEN, GPXQ_IN_FEATURES))
    # Build deterministic calibration data
    torch.manual_seed(GPXQ_SEED + 1)
    calib_input = torch.randn(GPXQ_NUM_SAMPLES, GPXQ_SEQLEN, GPXQ_IN_FEATURES)
    return model, calib_input


def _run_gptq_with_stats(model, calib_input, captured_hessians=None):
    """Run ``gptq_mode`` with a mocked ``_single_layer_update`` and return the
    collected :class:`DictStatsCollector`.

    The inner ``_single_layer_update`` is replaced by a no-op that is still
    wrapped with :func:`gpxq_stats_wrap`, so the statistics collection
    machinery (Hessian capture, pre/post error logging) executes normally
    while the actual GPTQ weight-update algorithm is skipped.

    If *captured_hessians* is a dict, the raw ``H`` tensors forwarded to the
    stat callbacks are stored under ``"pre_H"`` / ``"post_H"`` keys.
    """

    @gpxq_stats_wrap
    def _noop_single_layer_update(self):
        pass

    collector = DictStatsCollector()
    with collect_stats(collector):
        with patch.object(GPTQ, '_single_layer_update', _noop_single_layer_update):
            with torch.no_grad():
                with gptq_mode(model, use_quant_activations=False, create_weight_orig=True) as gptq:

                    # If the caller wants to capture raw Hessians, override
                    # the stat functions after gptq_mode registers its defaults
                    if captured_hessians is not None:

                        def _capture_hessian(prefix, name, layer, H, G=None, **kw):
                            captured_hessians[f"{prefix}_H"] = H.clone()
                            return gpxq_compute_error_stats(
                                prefix=prefix, name=name, layer=layer, H=H, G=G)

                        collector.on(
                            "pre_update", functools.partial(_capture_hessian, prefix="pre"))
                        collector.on(
                            "post_update", functools.partial(_capture_hessian, prefix="post"))

                    gptq_model = gptq.model
                    for _ in range(gptq.num_layers):
                        gptq_model(calib_input)
                        gptq.update()
    return collector


def _compute_expected_hessian(calib_input):
    """Replicate the iterative covariance formula from
    ``GPTQ.compute_iterative_covariance`` for a single batch fed to a
    Linear layer.

    For input ``X`` of shape ``[N, S, C]`` (batch, sequence, features),
    ``process_input`` flattens to ``[N*S, C]`` before computing the
    covariance, so the effective number of samples is ``N * S``::

        H = (2 / (N*S)) * X_flat^T X_flat     shape [1, C, C]
    """
    # Mirror process_input: reshape to 2-D then transpose
    X = calib_input.reshape(-1, calib_input.shape[-1]).to(torch.float32)  # [N*S, C]
    T = X.shape[0]  # effective sample count = N * S
    X = X.t().unsqueeze(0)  # [1, C, T]
    X = math.sqrt(2.0 / T) * X
    H = X.bmm(X.transpose(2, 1))  # [1, C, C]
    return H


# -- Test 1 ----------------------------------------------------------------


def test_gpxq_hessian_captured():
    """The Hessian tensor forwarded to the stats callback must equal the
    covariance matrix accumulated during calibration."""
    model, calib_input = _build_model_and_calibration_data()

    captured = {}
    _run_gptq_with_stats(model, calib_input, captured_hessians=captured)

    H_expected = _compute_expected_hessian(calib_input)
    assert "pre_H" in captured, "pre_update callback was never invoked"
    torch.testing.assert_close(captured["pre_H"], H_expected)
    # Since _single_layer_update is a no-op, pre and post use the same H
    torch.testing.assert_close(captured["post_H"], H_expected)


# -- Test 2 ----------------------------------------------------------------


def test_gpxq_error_stats():
    """The error statistics collected during GPTQ must match values computed
    by hand from the quantised weights, original weights and Hessian.

    The expected relative output error ``||X(Q-W)^T||_F / ||XW^T||_F`` is
    computed in two independent ways and cross-checked:

    * **From the inputs** (``X``): directly as the Frobenius norm ratio of
      the output error vs. the float output.
    * **From the Hessian** (``H``): using the identity
      ``||XE^T||_F^2 = tr(E H E^T)`` (up to a constant that cancels in the
      ratio), giving ``sqrt(tr(E H E^T) / tr(W H W^T))``.

    Because ``_single_layer_update`` is mocked to a no-op the weights are
    *not* modified by GPTQ, so pre- and post-update statistics are identical.
    """
    model, calib_input = _build_model_and_calibration_data()
    collector = _run_gptq_with_stats(model, calib_input)

    layer = model.linear
    layer_name = "linear"
    assert layer_name in collector.stats, (
        f"No stats collected for '{layer_name}'. Keys: {list(collector.stats.keys())}")

    stats = collector.stats[layer_name]

    # -- Compute expected values by hand --
    H = _compute_expected_hessian(calib_input).squeeze(0)  # [C, C]
    dtype = H.dtype
    Q = layer.quant_weight().value.to(dtype=dtype)  # [OC, C]
    W = layer.weight_orig.to(dtype=dtype, device=Q.device)  # [OC, C]
    err = Q - W

    expected_rel_weight_err = (torch.norm(err, p='fro') / torch.norm(W, p='fro')).item()

    # Path A – from the Hessian: sqrt(tr(E H E^T) / tr(W H W^T))
    rel_out_from_H = torch.sqrt(torch.trace(err @ H @ err.T) / torch.trace(W @ H @ W.T)).item()

    # Path B – from the calibration inputs: ||X E^T||_F / ||X W^T||_F
    X_flat = calib_input.reshape(-1, GPXQ_IN_FEATURES).to(dtype)
    rel_out_from_X = (torch.norm(X_flat @ err.T, p='fro') /
                      torch.norm(X_flat @ W.T, p='fro')).item()

    # Sanity-check: both paths must agree
    assert rel_out_from_H == pytest.approx(rel_out_from_X, abs=1e-5)

    # -- Assert pre-update stats --
    assert stats["pre_rel_weight_err"] == pytest.approx(expected_rel_weight_err, abs=1e-6)
    assert stats["pre_rel_out_err"] == pytest.approx(rel_out_from_H, abs=1e-5)

    # -- Assert post-update stats (identical because the mock is a no-op) --
    assert stats["post_rel_weight_err"] == pytest.approx(expected_rel_weight_err, abs=1e-6)
    assert stats["post_rel_out_err"] == pytest.approx(rel_out_from_H, abs=1e-5)
