"""
Copyright (c) 2016-     Facebook, Inc            (Adam Paszke)
Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala)
Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert)
Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu)
Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu)
Copyright (c) 2011-2013 NYU                      (Clement Farabet)
Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston)
Copyright (c) 2006      Idiap Research Institute (Samy Bengio)
Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)

All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

3. Neither the names of Xilinx, Facebook, Deepmind Technologies, NYU,
   NEC Laboratories America and IDIAP Research Institute nor the names
   of its contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.

Forked as-is from PyTorch 2.0.1
"""

from typing import Any, Dict, List, Tuple

try:
    from torch.utils._pytree import register_pytree_node
except:
    # Deprecated as of 2.3, but keeping for backportability
    from torch.utils._pytree import _register_pytree_node
    register_pytree_node = _register_pytree_node
from torch.utils._pytree import Context

from ._compatibility import compatibility

__all__ = ["immutable_list", "immutable_dict"]

_help_mutation = """\
If you are attempting to modify the kwargs or args of a torch.fx.Node object,
instead create a new copy of it and assign the copy to the node:
    new_args = ... # copy and mutate args
    node.args = new_args
"""


def _no_mutation(self, *args, **kwargs):
    raise NotImplementedError(
        f"'{type(self).__name__}' object does not support mutation. {_help_mutation}")


def _create_immutable_container(base, mutable_functions):
    container = type('immutable_' + base.__name__, (base,), {})
    for attr in mutable_functions:
        setattr(container, attr, _no_mutation)
    return container


immutable_list = _create_immutable_container(
    list,
    [
        '__delitem__',
        '__iadd__',
        '__imul__',
        '__setitem__',
        'append',
        'clear',
        'extend',
        'insert',
        'pop',
        'remove'])
immutable_list.__reduce__ = lambda self: (immutable_list, (tuple(iter(self)),))

compatibility(is_backward_compatible=True)(immutable_list)

immutable_dict = _create_immutable_container(
    dict, ['__delitem__', '__setitem__', 'clear', 'pop', 'popitem', 'update'])
immutable_dict.__reduce__ = lambda self: (immutable_dict, (iter(self.items()),))
compatibility(is_backward_compatible=True)(immutable_dict)

# Register immutable collections for PyTree operations


def _immutable_dict_flatten(d: Dict[Any, Any]) -> Tuple[List[Any], Context]:
    return list(d.values()), list(d.keys())


def _immutable_dict_unflatten(values: List[Any], context: Context) -> Dict[Any, Any]:
    return immutable_dict({key: value for key, value in zip(context, values)})


def _immutable_list_flatten(d: List[Any]) -> Tuple[List[Any], Context]:
    return d, None


def _immutable_list_unflatten(values: List[Any], context: Context) -> List[Any]:
    return immutable_list(values)


register_pytree_node(immutable_dict, _immutable_dict_flatten, _immutable_dict_unflatten)
register_pytree_node(immutable_list, _immutable_list_flatten, _immutable_list_unflatten)
