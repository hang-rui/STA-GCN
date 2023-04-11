import math
from torch import nn

from . import layers
from .nets import STAGCN


__activations = {
    'gelu': nn.GELU(),
    'relu': nn.ReLU(inplace=True),
    'relu6': nn.ReLU6(inplace=True),
    'Hswish': nn.Hardswish(inplace=True),
    'Hsigmoid': nn.Hardsigmoid(inplace=True),

}

def create(model_type, act_type, **kwargs):
    kwargs.update({
        'act': __activations[act_type]
    })
    return STAGCN(**kwargs)
