# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""
Beacon: Post-Training Quantization with Integrated Grid Selection

Implementation of the Beacon algorithm as proposed in:
    S. Zhang and R. Saab, "Beacon: Post-Training Quantization with Integrated Grid Selection,"
    IEEE Signal Processing Letters, 2026.
    https://arxiv.org/abs/2508.20293

Beacon operates on Brevitas quantized models (QuantLinear, QuantConv*d). It supports
two modes:

1. **With error correction** (default): Uses two distinct inputs X (float) and
   X_tilde (quantized) to account for quantization error propagation across layers.
   Requires two forward passes per batch (managed by gpfq_mode.catch_stopfwd).
   Sets L = R^{-T} @ G and L_tilde = R where R is Cholesky of H = X_tilde^T @ X_tilde.

2. **Without error correction**: Uses only X = X_tilde (same input for both),
   so L = L_tilde = R. Only requires one forward pass per batch, no G matrix needed.
   This is the recommended mode when combining Beacon with GPTQ/Qronos (scales_only),
   as described in the paper's GPTQ* variant.

After optimization, Beacon writes the computed per-channel scaling factors back into
the Brevitas quantizer infrastructure (scaling_impl.value).
"""

import math
from typing import List
from typing import Optional
import warnings

import torch
import torch.nn as nn

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gpxq import GPxQ
from brevitas.graph.gpxq import SUPPORTED_CONV_OP
from brevitas.graph.utils import is_conv_transposed
from brevitas.utils.torch_utils import StopFwdException


class BeaconLayerOptimizer(GPxQ):
    """
    Per-layer optimizer implementing the Beacon algorithm (Algorithm 1 from the paper).

    Beacon quantizes weights on a fixed unscaled integer grid A and determines
    optimal per-channel scaling factors c analytically by maximizing the cosine
    similarity between Lw and L_tilde @ q.

    Supports two variants:
    - With error correction (use_error_correction=True): L and L_tilde differ,
      computed from two distinct inputs (quantized and float).
    - Without error correction (use_error_correction=False): L = L_tilde = R
      (Cholesky of H), using a single input. Only H is needed.

    Inherits from GPxQ to reuse process_input (handles QuantTensor and input_quant),
    weight_orig management, and the hook infrastructure.

    Args:
        layer: The Brevitas QuantLinear or QuantConv*d layer to optimize.
        name: Name of the layer in the model.
        act_order: If True, process columns in descending order of H diagonal.
        len_parallel_layers: Number of layers being optimized in parallel.
        create_weight_orig: Whether to store original weights.
        bit_width: Number of bits for quantization (Beacon's own grid).
        num_loops: Number of cyclic coordinate descent refinement loops (ell_max).
        scales_only: If True, only update quantizer scales, do not modify weights.
        use_error_correction: If True, use two-pass collection (H and G).
            If False, use single-pass (H only, L = L_tilde = R).
        device: Device for buffers ('cpu' or 'same').
        dtype: Dtype for buffers.
    """

    def __init__(
            self,
            layer,
            name,
            act_order,
            len_parallel_layers,
            create_weight_orig,
            bit_width=4,
            num_loops=5,
            scales_only=False,
            use_error_correction=True,
            device='cpu',
            dtype=torch.float32) -> None:
        super().__init__(
            layer, name, act_order, len_parallel_layers, create_weight_orig, device, dtype)

        self.bit_width = bit_width
        self.num_loops = num_loops
        self.scales_only = scales_only
        self.use_error_correction = use_error_correction

        # H = X_tilde @ X_tilde^T (always needed)
        self.H = torch.zeros((self.groups, self.columns, self.columns),
                             device=self.device,
                             dtype=self.dtype)

        # G = X_tilde @ X^T (only needed with error correction)
        if self.use_error_correction:
            self.G = torch.zeros((self.groups, self.columns, self.columns),
                                 device=self.device,
                                 dtype=self.dtype)

        if self.use_intermediate_buffer:
            self.B = torch.zeros((self.groups, self.columns, self.columns),
                                 device=self.device,
                                 dtype=self.dtype,
                                 pin_memory=torch.cuda.is_available())

        self.quant_input = None

        # Build the fixed unscaled integer alphabet
        # A = {-2^(b-1), -2^(b-1)+1, ..., 0, ..., 2^(b-1)-1}
        half = 2 ** (bit_width - 1)
        self.alphabet = torch.arange(-half, half, 1.0, dtype=self.dtype)

        if bit_width > 6:
            warnings.warn(
                f"Beacon with bit_width={bit_width} uses an alphabet of size "
                f"{len(self.alphabet)}. This may be slow and memory-intensive "
                f"for large layers.")

    def update_batch(self, module, input, current_layer):
        """
        Hook callback for collecting covariance matrices.

        With error correction (two-pass, managed by gpfq_mode.catch_stopfwd):
            Pass 1 (weight_quant enabled): H += X_tilde @ X_tilde^T, store X_tilde
            Pass 2 (weight_quant disabled): G += X_tilde @ X^T, clear X_tilde

        Without error correction (single-pass):
            Only H += X @ X^T is collected. No G matrix needed.
        """
        if self.disable_pre_forward_hook:
            return input

        current_layer.layer_names.add(self.name)

        inp_processed = self.process_input(input)  # [groups, in_features, batch_size]
        batch_size = inp_processed.shape[-1]

        # Normalize for numerical stability (same as GPFQ)
        inp_processed = math.sqrt(1 / batch_size) * inp_processed.to(self.dtype)

        if self.use_error_correction:
            is_quant_enabled = module.weight_quant.is_quant_enabled
            if not is_quant_enabled:
                # Float pass: compute G = X_tilde @ X^T
                if self.use_intermediate_buffer:
                    self.B.copy_(self.quant_input.bmm(inp_processed.transpose(2, 1)))
                    self.G += self.B
                else:
                    self.G += self.quant_input.bmm(inp_processed.transpose(2, 1))
                self.quant_input = None
            else:
                # Quant pass: compute H = X_tilde @ X_tilde^T, store X_tilde
                if self.use_intermediate_buffer:
                    self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
                    self.H += self.B
                else:
                    self.H += inp_processed.bmm(inp_processed.transpose(2, 1))
                assert self.quant_input is None
                self.quant_input = inp_processed
        else:
            # No error correction: single pass, only collect H = X @ X^T
            if self.use_intermediate_buffer:
                self.B.copy_(inp_processed.bmm(inp_processed.transpose(2, 1)))
                self.H += self.B
            else:
                self.H += inp_processed.bmm(inp_processed.transpose(2, 1))

        current_layer.forward_count += 1
        if current_layer.forward_count == self.len_parallel_layers:
            current_layer.forward_count = 0
            raise StopFwdException

    def single_layer_update(self):
        """
        Beacon Algorithm 1: greedy initialization + cyclic coordinate descent.

        Quantizes each output channel's weight vector onto the fixed unscaled grid A,
        then computes the optimal per-channel scaling factor c analytically.
        Finally, writes the scaling factors into the Brevitas quantizer infrastructure.

        With error correction: L = R^{-T} @ G, L_tilde = R
        Without error correction: L = L_tilde = R
        """
        assert hasattr(self.layer, 'weight_orig'), \
            "Error: Beacon requires the original weights to be stored, see `create_weight_orig`."
        if hasattr(self.layer, 'allocate_params'):
            self.layer.allocate_params(self.layer)
        if self.use_intermediate_buffer:
            del self.B  # free memory

        weight = self.layer.weight.data
        weight_orig = self.layer.weight_orig.data
        dev = weight.device
        weight_orig = weight_orig.to(dev)
        dtype = weight.dtype

        # Reshape for conv layers
        if isinstance(self.layer, SUPPORTED_CONV_OP):
            if is_conv_transposed(self.layer):
                weight = weight.transpose(1, 0)
                weight_orig = weight_orig.transpose(1, 0)
            weight = weight.flatten(1)
            weight_orig = weight_orig.flatten(1)
        weight = weight.view(self.groups, -1, weight.shape[-1])
        weight_orig = weight_orig.view(self.groups, -1, weight_orig.shape[-1])

        N = self.columns
        alphabet = self.alphabet.to(dev, dtype=self.dtype)
        eps = 1e-10

        # Store per-channel scales across groups for writing back later
        all_scales = []

        for group_index in range(self.groups):
            H = self.H[group_index].to(dev, dtype=self.dtype)
            w = weight_orig[group_index].to(dev, dtype=self.dtype)  # [OC, N]

            OC = w.shape[0]

            # Handle dead channels
            dead = H.diag() == 0
            w[:, dead] = 0

            # Act order permutation
            if self.act_order:
                perm = torch.argsort(H.diag(), descending=True)
                H = H[perm][:, perm]
                w = w[:, perm]
            else:
                perm = torch.arange(N, device=dev)

            # Cholesky decomposition: R^T R = H (with damping)
            damp = 1e-6 * H.diag().max().clamp(min=1e-12)
            H_damped = H.clone()
            H_damped.diagonal().add_(damp)
            try:
                R = torch.linalg.cholesky(H_damped, upper=True)
            except Exception:
                warnings.warn(
                    f"Cholesky decomposition failed for layer {self.name} "
                    f"(group {group_index}). Skipping this group.")
                all_scales.append(torch.ones(OC, device=dev, dtype=self.dtype))
                continue

            L_tilde = R  # [N, N], upper triangular

            if self.use_error_correction:
                G = self.G[group_index].to(dev, dtype=self.dtype)
                if self.act_order:
                    G = G[perm][:, perm]
                # L = R^{-T} @ G
                L = torch.linalg.solve_triangular(R.T, G, upper=False)  # [N, N]
            else:
                # Without error correction: L = L_tilde = R
                L = R

            # Precompute full target: Lw = L @ w^T, shape [N, OC]
            Lw_full = L @ w.T
            Lw_full_norm = Lw_full.norm(dim=0, keepdim=True)  # [1, OC]

            # ===== Greedy initialization (Algorithm 1, lines 5-8) =====
            q = torch.zeros(OC, N, device=dev, dtype=self.dtype)
            # running_sum = L_tilde @ q^T, accumulated incrementally, shape [N, OC]
            running_sum = torch.zeros(N, OC, device=dev, dtype=self.dtype)

            for t in range(N):
                Lt_col = L_tilde[:t + 1, t]  # [t+1]

                # Partial target: L_{<=t} @ w_{<=t}^T
                target_t = L[:t + 1, :t + 1] @ w[:, :t + 1].T  # [t+1, OC]
                target_t_norm = target_t.norm(dim=0, keepdim=True)  # [1, OC]

                # Candidates: [|A|, t+1, OC]
                cand = running_sum[:t + 1].unsqueeze(0) + \
                    Lt_col.view(1, -1, 1) * alphabet.view(-1, 1, 1)

                dots = (target_t.unsqueeze(0) * cand).sum(dim=1)  # [|A|, OC]
                norms_c = cand.norm(dim=1)  # [|A|, OC]
                cos_sims = dots / (target_t_norm * norms_c + eps)  # [|A|, OC]

                best_idx = cos_sims.argmax(dim=0)  # [OC]
                best_p = alphabet[best_idx]  # [OC]

                q[:, t] = best_p
                running_sum[:t + 1] += Lt_col.unsqueeze(1) * best_p.unsqueeze(0)

            # Recompute running_sum exactly to avoid accumulated numerical errors
            running_sum = L_tilde @ q.T  # [N, OC]

            # ===== Refinement loops (Algorithm 1, lines 9-13) =====
            for loop_idx in range(self.num_loops):
                for t in range(N):
                    Lt_col = L_tilde[:, t]  # [N]

                    # Remove current q_t contribution
                    running_sum -= Lt_col.unsqueeze(1) * q[:, t].unsqueeze(0)

                    # Candidates: [|A|, N, OC]
                    cand = running_sum.unsqueeze(0) + \
                        Lt_col.view(1, -1, 1) * alphabet.view(-1, 1, 1)

                    dots = (Lw_full.unsqueeze(0) * cand).sum(dim=1)  # [|A|, OC]
                    norms_c = cand.norm(dim=1)  # [|A|, OC]
                    cos_sims = dots / (Lw_full_norm * norms_c + eps)  # [|A|, OC]

                    best_idx = cos_sims.argmax(dim=0)  # [OC]
                    best_p = alphabet[best_idx]  # [OC]

                    q[:, t] = best_p
                    running_sum += Lt_col.unsqueeze(1) * best_p.unsqueeze(0)

            # ===== Compute optimal scale per output channel (Algorithm 1, line 14) =====
            # c = <Lw, L_tilde @ q> / ||L_tilde @ q||^2
            Ltq = L_tilde @ q.T  # [N, OC]
            numerator = (Lw_full * Ltq).sum(dim=0)  # [OC]
            denominator = (Ltq * Ltq).sum(dim=0)  # [OC]
            c = numerator / (denominator + eps)  # [OC]

            # Final quantized weights
            final_w = c.unsqueeze(1) * q  # [OC, N]

            # Undo permutation
            if self.act_order:
                inv_perm = torch.argsort(perm)
                final_w = final_w[:, inv_perm]

            if not self.scales_only:
                weight[group_index] = final_w.to(dtype)
            all_scales.append(c)

        # Write the computed scales into Brevitas quantizer infrastructure
        self._write_scales(all_scales, final_w)

        del self.H
        if self.use_error_correction:
            del self.G

        if hasattr(self.layer, 'offload_params'):
            self.layer.offload_params(self.layer)

    def _write_scales(self, all_scales, final_w):
        """
        Write the Beacon-computed per-channel scaling factors into the Brevitas
        quantizer's scaling_impl.value parameter.

        The Brevitas scaling pipeline computes:
            effective_scale = restrict_clamp_scaling(scaling_impl.value) / int_scaling_impl(bit_width)

        int_scaling_impl computes the integer threshold (e.g., 2^(b-1) - 1 for symmetric).

        Beacon's c is the full scale factor, so we need to store:
            scaling_impl.value = c * int_threshold
        where int_threshold is determined by Brevitas's int_scaling_impl.

        Beacon requires signed_float_scale (SignedFloatRestrictValue) so that
        negative scale factors are preserved. This is enforced by the CLI
        validation in llm_args.py.
        """
        try:
            scaling_impl = self.layer.weight_quant.tensor_quant.scaling_impl
        except AttributeError:
            warnings.warn(
                f"Could not access scaling_impl for layer {self.name}. "
                f"Scales will not be written back to quantizer.")
            return

        # Concatenate scales from all groups: [total_OC]
        c_all = torch.cat(all_scales, dim=0)

        # Get int_threshold from the quantizer's int_scaling_impl
        try:
            bit_width_tensor = self.layer.weight_quant.tensor_quant.msb_clamp_bit_width_impl()
            int_threshold = self.layer.weight_quant.tensor_quant.int_scaling_impl(bit_width_tensor)
        except Exception:
            # Fallback: compute manually for symmetric int quant
            int_threshold = 2 ** (self.bit_width - 1) - 1
            warnings.warn(
                f"Could not compute int_threshold from quantizer for {self.name}. "
                f"Using manual value: {int_threshold}")

        # Store c * int_threshold into scaling_impl.value
        # Shape: [out_channels, 1] for per-channel Linear
        new_value = (c_all * int_threshold).unsqueeze(-1)

        if hasattr(scaling_impl, 'value') and isinstance(scaling_impl.value, nn.Parameter):
            scaling_impl.value.data.copy_(
                new_value.to(scaling_impl.value.device, dtype=scaling_impl.value.dtype))
            # Mark scaling as initialized
            if hasattr(scaling_impl, 'init_done'):
                scaling_impl.init_done = True
        else:
            warnings.warn(
                f"scaling_impl for {self.name} does not have a 'value' Parameter. "
                f"Scales will not be written back.")


class beacon_mode(gpfq_mode):
    """
    Context manager for applying the Beacon PTQ algorithm to a Brevitas quantized model.

    Inherits from gpfq_mode to reuse:
        - quantization_status_manager (disables quantization during forward passes)
        - _is_module_supported (checks for QuantLinear/QuantConv*d with weight_quant enabled)
        - Layer discovery and hook registration

    Supports two modes:
    - With error correction (default): Uses gpfq_mode's two-pass catch_stopfwd.
    - Without error correction: Uses single-pass catch_stopfwd (only collects H).

    Args:
        model: Brevitas quantized model (QuantLinear/QuantConv*d layers).
        group_of_parallel_layers: Groups of layers to optimize in parallel.
        inplace: Whether to apply in place or deepcopy. Default: True.
        create_weight_orig: Store original weights. Default: True.
        use_quant_activations: Use quantized activations for H. Default: True.
        return_forward_output: Whether to return forward output. Default: False.
        act_order: Whether to order columns by activation magnitude. Default: False.
        bit_width: Number of bits for Beacon quantization grid. Default: 4.
        num_loops: Number of cyclic refinement loops. Default: 5.
        scales_only: If True, only update scales, not weights. Default: False.
        use_error_correction: If True, two-pass mode (H and G). If False,
            single-pass mode (H only, L = L_tilde = R). Default: True.
        device: Device for buffers. Default: 'cpu'.
        dtype: Dtype for buffers. Default: torch.float32.

    Example:
        >>> # With error correction (default)
        >>> with torch.no_grad():
        >>>     with beacon_mode(model, bit_width=4, num_loops=5) as bcn:
        >>>         for i in range(bcn.num_layers):
        >>>             for images, _ in calib_loader:
        >>>                 bcn.model(images)
        >>>             bcn.update()
        >>>
        >>> # Without error correction (scales only, for combining with GPTQ)
        >>> with torch.no_grad():
        >>>     with beacon_mode(model, scales_only=True, use_error_correction=False) as bcn:
        >>>         for i in range(bcn.num_layers):
        >>>             for images, _ in calib_loader:
        >>>                 bcn.model(images)
        >>>             bcn.update()
    """

    def __init__(
            self,
            model: nn.Module,
            group_of_parallel_layers: Optional[List[str]] = None,
            inplace: bool = True,
            create_weight_orig: bool = True,
            use_quant_activations: bool = True,
            return_forward_output: bool = False,
            act_order: bool = False,
            bit_width: int = 4,
            num_loops: int = 5,
            scales_only: bool = False,
            use_error_correction: bool = True,
            device: str = 'cpu',
            dtype: torch.dtype = torch.float32) -> None:
        self.bit_width = bit_width
        self.num_loops = num_loops
        self.scales_only = scales_only
        self.use_error_correction = use_error_correction
        # gpfq_mode.__init__ will call gpxq_mode.__init__ which calls
        # quantization_status_manager.__init__ and sets up the model
        super().__init__(
            model=model,
            group_of_parallel_layers=group_of_parallel_layers,
            inplace=inplace,
            create_weight_orig=create_weight_orig,
            use_quant_activations=use_quant_activations,
            return_forward_output=return_forward_output,
            act_order=act_order,
            algorithm_impl=BeaconLayerOptimizer,  # Not used directly; overridden below
            device=device,
            dtype=dtype)

    def catch_stopfwd(self, *args, **kwargs):
        if self.use_error_correction:
            # Two-pass: inherited from gpfq_mode
            super().catch_stopfwd(*args, **kwargs)
        else:
            # Single-pass: only collect H (no G needed)
            # Run with quantization disabled so we get float inputs
            with quantization_status_manager(
                    self.model,
                    disable_act_quant=True,
                    disable_weight_quant=True,
                    disable_bias_quant=True,
                    is_training=False,
            ):
                try:
                    self.orig_forward(*args, **kwargs)
                except StopFwdException:
                    pass

            if self.return_forward_output:
                for name, gpxq_class in self.gpxq_layers.items():
                    gpxq_class.disable_pre_forward_hook = True
                out = self.orig_forward(*args, **kwargs)
                for name, gpxq_class in self.gpxq_layers.items():
                    gpxq_class.disable_pre_forward_hook = False
                return out

    def initialize_module_optimizer(self, layer, name, len_parallel_layers, create_weight_orig):
        return BeaconLayerOptimizer(
            layer=layer,
            name=name,
            act_order=self.act_order,
            len_parallel_layers=len_parallel_layers,
            create_weight_orig=create_weight_orig,
            bit_width=self.bit_width,
            num_loops=self.num_loops,
            scales_only=self.scales_only,
            use_error_correction=self.use_error_correction,
            device=self.device,
            dtype=self.dtype)
