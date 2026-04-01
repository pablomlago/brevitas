# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from copy import deepcopy
from functools import partial

from accelerate.utils.operations import send_to_device
import torch
from tqdm import tqdm

from brevitas.graph.calibrate import quantization_status_manager
from brevitas.graph.gpfq import GPFQ
from brevitas.graph.gpfq import gpfq_mode
from brevitas.graph.gptq import GPTQ
from brevitas.graph.gptq import gptq_mode
from brevitas.graph.magr import magr_mode
from brevitas.graph.qronos import Qronos
from brevitas.utils.python_utils import recurse_getattr
from brevitas.utils.torch_utils import StopFwdException
from brevitas_examples.common.axe import A2GPFQ
from brevitas_examples.common.axe import A2GPTQ


def _get_block_device(block):
    """Get the device of a block from its first parameter."""
    return next(block.parameters()).device


def _gpxq_block_optimization_callback(block, gpxq, cached_args, cached_kwargs):
    device = _get_block_device(block)
    for _ in tqdm(range(gpxq.num_layers), desc="Layers", leave=False):
        for args, kwargs in zip(cached_args, cached_kwargs):
            args = send_to_device(args, device)
            kwargs = send_to_device(kwargs, device)
            block(*args, **kwargs)
        gpxq.update()


def _magr_block_optimization_callback(block, magr, cached_args, cached_kwargs):
    device = _get_block_device(block)
    for args, kwargs in zip(cached_args, cached_kwargs):
        args = send_to_device(args, device)
        kwargs = send_to_device(kwargs, device)
        block(*args, **kwargs)
    magr.update()


@torch.no_grad()
def block_optimization(
        model,
        dataloader,
        block_name,
        context_manager_func,
        context_manager_kwargs,
        block_optimization_callback=_gpxq_block_optimization_callback):
    disable_quantization_cm = quantization_status_manager(
        model=model,
        disable_act_quant=not context_manager_kwargs.get('use_quant_activations', True),
        disable_weight_quant=False,
        disable_bias_quant=not context_manager_kwargs.get('use_quant_activations', True),
    )
    cache_state = model.config.use_cache
    model.config.use_cache = False
    blocks = recurse_getattr(model, block_name)
    first_block = blocks[0]
    cached_args, cached_kwargs = [], []

    # Intercept input to first block
    def intercept_input(module, args, kwargs):
        args = send_to_device(args, 'cpu')
        kwargs = send_to_device(kwargs, 'cpu')
        cached_args.append(args)
        cached_kwargs.append(kwargs)
        raise StopFwdException

    # Intercept output from block N-1 to set it as input to block N
    def intercept_output(module, args, kwargs, output):
        if isinstance(output, tuple):
            output = output[0]
        output = send_to_device(output, 'cpu')
        cached_args.append((output,))
        raise StopFwdException

    # Collect input to first block
    hook = first_block.register_forward_pre_hook(intercept_input, with_kwargs=True)
    with disable_quantization_cm:
        for inps in dataloader:
            try:
                model(**inps)
            except StopFwdException:
                pass
    hook.remove()

    # Iterate through all the blocks
    for index, block in tqdm(enumerate(blocks), desc="Blocks", total=len(blocks)):
        with context_manager_func(block, **context_manager_kwargs) as gpxq:
            block_optimization_callback(block, gpxq, cached_args, cached_kwargs)

        if index < len(blocks) - 1:
            # Once the block is done, we need to update the input to the next block
            past_cached_args, past_cached_kwargs = deepcopy(cached_args), deepcopy(cached_kwargs)
            cached_args = []
            hook = block.register_forward_hook(intercept_output, with_kwargs=True)
            device = _get_block_device(block)

            with disable_quantization_cm:
                for args, kwargs in zip(past_cached_args, past_cached_kwargs):
                    try:
                        args = send_to_device(args, device)
                        kwargs = send_to_device(kwargs, device)
                        block(*args, **kwargs)
                    except StopFwdException:
                        pass
            hook.remove()
    # Restore cache state
    model.config.use_cache = cache_state


def _dual_block_optimization_callback(
        block, gpxq, cached_args_quant, cached_kwargs_quant, cached_args_float,
        cached_kwargs_float):
    """Block optimization callback for layerwise-equivalent GPFQ/Qronos.

    Feeds quantized cached inputs for the quant pass and float cached inputs
    for the float pass, matching the activation pattern of layerwise mode.
    Requires gpfq_mode to be in single_pass_mode=True.
    """
    device = _get_block_device(block)
    for _ in tqdm(range(gpxq.num_layers), desc="Layers", leave=False):
        for args_q, kwargs_q, args_f, kwargs_f in zip(
                cached_args_quant, cached_kwargs_quant, cached_args_float,
                cached_kwargs_float):
            # Pass 1: quant enabled (default state from gpfq_mode)
            args_q = send_to_device(args_q, device)
            kwargs_q = send_to_device(kwargs_q, device)
            block(*args_q, **kwargs_q)
            # Pass 2: quant disabled
            with quantization_status_manager(gpxq.model,
                                             disable_act_quant=True,
                                             disable_weight_quant=True,
                                             disable_bias_quant=True):
                args_f = send_to_device(args_f, device)
                kwargs_f = send_to_device(kwargs_f, device)
                block(*args_f, **kwargs_f)
        gpxq.update()


@torch.no_grad()
def block_optimization_layerwise(
        model, dataloader, block_name, context_manager_func, context_manager_kwargs):
    """Block optimization with layerwise-equivalent activation handling for GPFQ/Qronos.

    Unlike standard block_optimization which caches a single set of activations per block
    boundary (causing both the quant and float passes to see the same input), this function
    maintains two separate sets of cached activations:
      - Quantized activations: block outputs with quantization enabled
      - Float activations: block outputs with quantization fully disabled

    This ensures each layer sees the same activations it would in layerwise mode, where the
    quant pass propagates through all preceding quantized layers and the float pass propagates
    through all preceding layers with quantization disabled.
    """
    cache_state = model.config.use_cache
    model.config.use_cache = False
    blocks = recurse_getattr(model, block_name)
    first_block = blocks[0]

    # Two separate caches for quantized and float activations
    cached_args_quant, cached_kwargs_quant = [], []
    cached_args_float, cached_kwargs_float = [], []

    # Intercept input to first block - stores into whichever list is set as target
    def intercept_input(module, args, kwargs):
        args = send_to_device(args, 'cpu')
        kwargs = send_to_device(kwargs, 'cpu')
        intercept_input.target_args.append(args)
        intercept_input.target_kwargs.append(kwargs)
        raise StopFwdException

    # Intercept output from block N-1 to set it as input to block N
    def intercept_output(module, args, kwargs, output):
        if isinstance(output, tuple):
            output = output[0]
        output = send_to_device(output, 'cpu')
        intercept_output.target_args.append((output,))
        raise StopFwdException

    # Collect quantized inputs to first block (with model's default quant state)
    hook = first_block.register_forward_pre_hook(intercept_input, with_kwargs=True)

    intercept_input.target_args = cached_args_quant
    intercept_input.target_kwargs = cached_kwargs_quant
    for inps in dataloader:
        try:
            model(**inps)
        except StopFwdException:
            pass

    # Collect float inputs to first block (all quantization disabled)
    intercept_input.target_args = cached_args_float
    intercept_input.target_kwargs = cached_kwargs_float
    with quantization_status_manager(model,
                                     disable_act_quant=True,
                                     disable_weight_quant=True,
                                     disable_bias_quant=True):
        for inps in dataloader:
            try:
                model(**inps)
            except StopFwdException:
                pass

    hook.remove()

    # Ensure single_pass_mode is enabled so the callback controls quant/float passes
    context_manager_kwargs = dict(context_manager_kwargs)
    context_manager_kwargs['single_pass_mode'] = True

    # Iterate through all the blocks
    for index, block in tqdm(enumerate(blocks), desc="Blocks", total=len(blocks)):
        with context_manager_func(block, **context_manager_kwargs) as gpxq:
            _dual_block_optimization_callback(
                block,
                gpxq,
                cached_args_quant,
                cached_kwargs_quant,
                cached_args_float,
                cached_kwargs_float)

        if index < len(blocks) - 1:
            # Capture two sets of outputs for the next block
            past_cached_args_quant = deepcopy(cached_args_quant)
            past_cached_kwargs_quant = deepcopy(cached_kwargs_quant)
            past_cached_args_float = deepcopy(cached_args_float)
            past_cached_kwargs_float = deepcopy(cached_kwargs_float)
            device = _get_block_device(block)

            # Quantized outputs for the next block
            cached_args_quant = []
            hook = block.register_forward_hook(intercept_output, with_kwargs=True)
            intercept_output.target_args = cached_args_quant
            for args, kwargs in zip(past_cached_args_quant, past_cached_kwargs_quant):
                try:
                    args = send_to_device(args, device)
                    kwargs = send_to_device(kwargs, device)
                    block(*args, **kwargs)
                except StopFwdException:
                    pass
            hook.remove()

            # Float outputs for the next block (all quantization disabled)
            cached_args_float = []
            hook = block.register_forward_hook(intercept_output, with_kwargs=True)
            intercept_output.target_args = cached_args_float
            with quantization_status_manager(block,
                                             disable_act_quant=True,
                                             disable_weight_quant=True,
                                             disable_bias_quant=True):
                for args, kwargs in zip(past_cached_args_float, past_cached_kwargs_float):
                    try:
                        args = send_to_device(args, device)
                        kwargs = send_to_device(kwargs, device)
                        block(*args, **kwargs)
                    except StopFwdException:
                        pass
            hook.remove()

            # kwargs (attention_mask, position_ids, etc.) don't change between blocks
            cached_kwargs_quant = past_cached_kwargs_quant
            cached_kwargs_float = past_cached_kwargs_float

    # Restore cache state
    model.config.use_cache = cache_state


@torch.no_grad()
def apply_gptq(
        model,
        dataloader,
        act_order=True,
        use_quant_activations=False,
        create_weight_orig=False,
        group_of_parallel_layers=None,
        block_name=None,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=None,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        gptq_class = partial(
            A2GPTQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        gptq_class = GPTQ
    if block_name is not None:
        context_manager_kwargs = {
            'act_order': act_order,
            'group_of_parallel_layers': group_of_parallel_layers,
            'create_weight_orig': create_weight_orig,
            'use_quant_activations': use_quant_activations,
            'gptq_class': gptq_class,
            'device': buffer_device,
            'dtype': buffer_dtype}
        block_optimization(model, dataloader, block_name, gptq_mode, context_manager_kwargs)
    else:
        with gptq_mode(model,
                       use_quant_activations=use_quant_activations,
                       group_of_parallel_layers=group_of_parallel_layers,
                       act_order=act_order,
                       create_weight_orig=create_weight_orig,
                       gptq_class=gptq_class,
                       device=buffer_device,
                       dtype=buffer_dtype) as gptq:
            gptq_model = gptq.model
            for _ in tqdm(range(gptq.num_layers)):
                for inps in dataloader:
                    gptq_model(**inps)
                gptq.update()


def _dual_optimization_callback(
        model,
        dataloader,
        act_order=True,
        block_name=None,
        group_of_parallel_layers=None,
        algorithm_impl=GPFQ,
        layerwise=False,
        device='cpu',
        dtype=torch.float32):
    """
    This wraps gpfq_mode, which can be used for any layerwise PTQ algorithm that
    optimizes the mismatched objective function || XW - \tilde{X}Q ||, where
    Q is the quantized weights and \tilde{X} are the (potentially quantized)
    activations resulting from the previously quantized layers.

    See https://arxiv.org/abs/2505.11695 for more!
    """
    if block_name is not None:
        context_manager_kwargs = {
            'act_order': act_order,
            'group_of_parallel_layers': group_of_parallel_layers,
            'create_weight_orig': True,
            'algorithm_impl': algorithm_impl,
            'device': device,
            'dtype': dtype}
        if layerwise:
            block_optimization_layerwise(
                model, dataloader, block_name, gpfq_mode, context_manager_kwargs)
        else:
            block_optimization(model, dataloader, block_name, gpfq_mode, context_manager_kwargs)
    else:
        with gpfq_mode(model,
                       act_order=act_order,
                       group_of_parallel_layers=group_of_parallel_layers,
                       create_weight_orig=True,
                       algorithm_impl=algorithm_impl,
                       device=device,
                       dtype=dtype) as algo:
            algo_model = algo.model
            for _ in tqdm(range(algo.num_layers)):
                for inps in dataloader:
                    algo_model(**inps)
                algo.update()


@torch.no_grad()
def apply_gpfq(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        block_name=None,
        max_accumulator_bit_width=None,
        max_accumulator_tile_size=None,
        layerwise=False,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    if max_accumulator_bit_width is not None:
        # Use accumulator-aware extension (AXE) framework
        print(f"Using AXE to target {max_accumulator_bit_width}-bit accumulation...")
        algorithm_impl = partial(
            A2GPFQ,
            max_accumulator_bit_width=max_accumulator_bit_width,
            max_accumulator_tile_size=max_accumulator_tile_size)
    else:
        algorithm_impl = GPFQ
    # We use the dual optimization callback, which uses two forward passes to correct
    # quantization error in both the weights and activations from previous layers
    _dual_optimization_callback(
        model,
        dataloader,
        act_order=act_order,
        block_name=block_name,
        group_of_parallel_layers=group_of_parallel_layers,
        algorithm_impl=algorithm_impl,
        layerwise=layerwise,
        device=buffer_device,
        dtype=buffer_dtype)


@torch.no_grad()
def apply_qronos(
        model,
        dataloader,
        act_order=True,
        group_of_parallel_layers=None,
        block_name=None,
        alpha=1e-6,
        layerwise=False,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    assert alpha > 0, "Error: alpha needs to be strictly positive"
    # We use the dual optimization callback, which uses two forward passes to correct
    # quantization error in both the weights and activations from previous layers
    _dual_optimization_callback(
        model,
        dataloader,
        act_order=act_order,
        block_name=block_name,
        group_of_parallel_layers=group_of_parallel_layers,
        algorithm_impl=partial(Qronos, alpha=alpha),
        layerwise=layerwise,
        device=buffer_device,
        dtype=buffer_dtype)


@torch.no_grad()
def apply_magr(
        model,
        dataloader,
        create_weight_orig=False,
        group_of_parallel_layers=None,
        block_name=None,
        alpha=0.01,
        num_steps=200,
        buffer_device='cpu',
        buffer_dtype=torch.float32):
    if block_name is not None:
        context_manager_kwargs = {
            'group_of_parallel_layers': group_of_parallel_layers,
            'create_weight_orig': create_weight_orig,
            'alpha': alpha,
            'num_steps': num_steps,
            'device': buffer_device,
            'dtype': buffer_dtype}
        block_optimization(
            model,
            dataloader,
            block_name,
            magr_mode,
            context_manager_kwargs,
            block_optimization_callback=_magr_block_optimization_callback)
    else:
        with magr_mode(model,
                       group_of_parallel_layers=group_of_parallel_layers,
                       create_weight_orig=create_weight_orig,
                       num_steps=num_steps,
                       alpha=alpha,
                       device=buffer_device,
                       dtype=buffer_dtype) as magr:
            magr_model = magr.model
            for inps in tqdm(dataloader, desc="Calculating covariances..."):
                magr_model(**inps)
            magr.update()
