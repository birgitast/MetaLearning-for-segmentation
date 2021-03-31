import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from torchmeta.modules.module import MetaModule


class MetaConvTranspose2d(nn.ConvTranspose2d, MetaModule):
    """Applies a 2D transposed convolution operator over an input image
    composed of several input planes.
    Adapted from PyTorch's ConvTranspose2D.
    """

    __doc__ = nn.ConvTranspose2d.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())
        bias = params.get('bias', None)

        return F.conv_transpose2d(input, params['weight'], bias, self.stride,
                        self.padding, self.output_padding, self.groups, self.dilation)