# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from typing import Optional, Tuple, Type, Union

from packaging import version
import torch
from torch import Tensor
from torch.nn import ConvTranspose1d
from torch.nn import ConvTranspose2d
from torch.nn import ConvTranspose3d
from torch.nn.functional import conv_transpose1d
from torch.nn.functional import conv_transpose2d
from torch.nn.functional import conv_transpose3d

from brevitas import torch_version
from brevitas.function.ops import max_int
from brevitas.function.ops_ste import ceil_ste
from brevitas.inject.defaults import Int8WeightPerTensorFloat
from brevitas.quant_tensor import QuantTensor

from .quant_layer import ActQuantType
from .quant_layer import BiasQuantType
from .quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from .quant_layer import WeightQuantType

__all__ = ['QuantConvTranspose1d', 'QuantConvTranspose2d', 'QuantConvTranspose3d']


class QuantConvTranspose1d(QuantWBIOL, ConvTranspose1d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            stride: int = 1,
            padding: int = 0,
            output_padding: int = 0,
            dilation: int = 1,
            groups: int = 1,
            padding_mode: str = 'zeros',
            bias: Optional[bool] = True,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs) -> None:
        ConvTranspose1d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
            device=device,
            dtype=dtype)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self._output_size = None

    @property
    def per_elem_ops(self):
        raise NotImplementedError

    @property
    def output_channel_dim(self) -> int:
        return 1

    @property
    def channelwise_separable(self) -> bool:
        raise self.groups == self.out_channels

    def forward(self,
                input: Union[Tensor, QuantTensor],
                output_size=None) -> Union[Tensor, QuantTensor]:
        self._output_size = output_size  # cache the value temporarily
        return self.forward_impl(input)

    def compute_output_padding(self, inp, output_size):
        if torch_version >= version.parse('1.12'):
            return self._output_padding(
                inp, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims=1)
        else:
            return self._output_padding(
                inp, output_size, self.stride, self.padding, self.kernel_size)

    def conv_transpose1d_zeros_pad(
            self, x: Tensor, weight: Tensor, bias: Optional[Tensor], output_padding):
        out = conv_transpose1d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
            dilation=self.dilation)
        return out

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_mode == 'zeros':
            output_padding = self.compute_output_padding(x, self._output_size)
            self._output_size = None  # set it back to None after consuming it
            out = self.conv_transpose1d_zeros_pad(x, quant_weight, quant_bias, output_padding)
            return out
        else:
            raise NotImplementedError(f"Padding mode {self.padding_mode} not supported.")


class QuantConvTranspose2d(QuantWBIOL, ConvTranspose2d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]],
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = 0,
            output_padding: Union[int, Tuple[int]] = 0,
            dilation: Union[int, Tuple[int]] = 1,
            groups: int = 1,
            padding_mode: str = 'zeros',
            bias: Optional[bool] = True,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs) -> None:
        ConvTranspose2d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
            device=device,
            dtype=dtype)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self._output_size = None

    @property
    def per_elem_ops(self):
        raise NotImplementedError

    @property
    def output_channel_dim(self) -> int:
        return 1

    @property
    def channelwise_separable(self) -> bool:
        raise self.groups == self.out_channels

    def forward(self,
                input: Union[Tensor, QuantTensor],
                output_size=None) -> Union[Tensor, QuantTensor]:
        self._output_size = output_size  # cache the value temporarily
        return self.forward_impl(input)

    def compute_output_padding(self, inp, output_size):
        if torch_version >= version.parse('1.12'):
            return self._output_padding(
                inp, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims=2)
        else:
            return self._output_padding(
                inp, output_size, self.stride, self.padding, self.kernel_size)

    def conv_transpose2d_zeros_pad(
            self, x: Tensor, weight: Tensor, bias: Optional[Tensor], output_padding):
        out = conv_transpose2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
            dilation=self.dilation)
        return out

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_mode == 'zeros':
            output_padding = self.compute_output_padding(x, self._output_size)
            self._output_size = None  # set it back to None after consuming it
            out = self.conv_transpose2d_zeros_pad(x, quant_weight, quant_bias, output_padding)
            return out
        else:
            raise NotImplementedError(f"Padding mode {self.padding_mode} not supported.")


class QuantConvTranspose3d(QuantWBIOL, ConvTranspose3d):

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[int, Tuple[int]],
            stride: Union[int, Tuple[int]] = 1,
            padding: Union[int, Tuple[int]] = 0,
            output_padding: Union[int, Tuple[int]] = 0,
            dilation: Union[int, Tuple[int]] = 1,
            groups: int = 1,
            padding_mode: str = 'zeros',
            bias: bool = True,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            input_quant: Optional[ActQuantType] = None,
            output_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            **kwargs) -> None:
        ConvTranspose3d.__init__(
            self,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            groups=groups,
            padding_mode=padding_mode,
            bias=bias,
            device=device,
            dtype=dtype)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=input_quant,
            output_quant=output_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        self._output_size = None

    @property
    def per_elem_ops(self):
        raise NotImplementedError

    @property
    def output_channel_dim(self) -> int:
        return 1

    @property
    def channelwise_separable(self) -> bool:
        raise self.groups == self.out_channels

    def forward(self,
                input: Union[Tensor, QuantTensor],
                output_size=None) -> Union[Tensor, QuantTensor]:
        self._output_size = output_size  # cache the value temporarily
        return self.forward_impl(input)

    def compute_output_padding(self, inp, output_size):
        if torch_version >= version.parse('1.12'):
            return self._output_padding(
                inp, output_size, self.stride, self.padding, self.kernel_size, num_spatial_dims=3)
        else:
            return self._output_padding(
                inp, output_size, self.stride, self.padding, self.kernel_size)

    def conv_transpose3d_zeros_pad(
            self, x: Tensor, weight: Tensor, bias: Optional[Tensor], output_padding):
        out = conv_transpose3d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            output_padding=output_padding,
            groups=self.groups,
            dilation=self.dilation)
        return out

    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        if self.padding_mode == 'zeros':
            output_padding = self.compute_output_padding(x, self._output_size)
            self._output_size = None  # set it back to None after consuming it
            out = self.conv_transpose3d_zeros_pad(x, quant_weight, quant_bias, output_padding)
            return out
        else:
            raise NotImplementedError(f"Padding mode {self.padding_mode} not supported.")
