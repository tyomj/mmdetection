import torch
from mmcv.cnn import CONV_LAYERS
from torch import nn
from torch.nn.modules.utils import _ntuple


class _NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return _NewEmptyTensorOp.apply(grad, shape), None


@CONV_LAYERS.register_module('ConvWC2d')
class ConvWC2d(nn.Conv2d):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=False):
        super(ConvWC2d, self).__init__(in_channels, out_channels, kernel_size,
                                       stride, padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(
            dim=1, keepdim=True).mean(
                dim=2, keepdim=True).mean(
                    dim=3, keepdim=True)
        weight = weight - weight_mean
        if x.numel() > 0:
            return nn.functional.conv2d(x, weight, self.bias, self.stride,
                                        self.padding, self.dilation,
                                        self.groups)
        _pair = _ntuple(2)
        output_shape = [
            (i + 2 * p - (di * (k - 1) + 1)) // d + 1
            for i, p, di, k, d in zip(x.shape[-2:], _pair(
                self.padding), _pair(self.dilation), _pair(self.kernel_size),
                                      _pair(self.stride))
        ]
        output_shape = [x.shape[0], self.weight.shape[0]] + output_shape
        return _NewEmptyTensorOp.apply(x, output_shape)
