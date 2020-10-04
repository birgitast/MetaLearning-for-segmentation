import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
from collections import OrderedDict

import math


# simple UNet implementation
class Unet(MetaModule):
    def __init__(
        self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True
    ):
        super(Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        #filters = [64, 128, 256, 512, 1024, 2048, 4096]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        """
        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv6 = unetConv2(filters[4], filters[5], self.is_batchnorm)
        self.maxpool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        """

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm)

        # upsampling
        """
        self.up_concat6 = unetUp(filters[6], filters[5], self.is_deconv)
        self.up_concat5 = unetUp(filters[5], filters[4], self.is_deconv)
        """

        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)




        # final conv (without any concat)
        self.final = MetaConv2d(filters[0], n_classes, kernel_size=1)


        """for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    print("conv")
                    print(m)
                    nn.init.xavier_uniform(m.weight)
                    nn.init.xavier_uniform(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    print("batch")
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()"""

    def forward(self, inputs, params=None):
        
        conv1 = self.conv1(inputs, params=self.get_subdict(params, 'conv1.double_conv'))
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1, params=self.get_subdict(params, 'conv2.double_conv'))
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2, params=self.get_subdict(params, 'conv3.double_conv'))
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3, params=self.get_subdict(params, 'conv4.double_conv'))
        maxpool4 = self.maxpool4(conv4)

        """
        conv5 = self.conv5(maxpool4)
        maxpool5 = self.maxpool5(conv5)
        
        conv6 = self.conv6(maxpool5)
        maxpool6 = self.maxpool6(conv6)
        
        test = maxpool6[0][0].detach().unsqueeze(0)

        """

        center = self.center(maxpool4, params=self.get_subdict(params, 'center.double_conv'))
        
        """
        up6 = self.up_concat6(conv6, center)
        up5 = self.up_concat5(conv5, up6)
        """

        up4 = self.up_concat4(conv4, center, params=self.get_subdict(params, 'up_concat4.conv.double_conv'))
        up3 = self.up_concat3(conv3, up4, params=self.get_subdict(params, 'up_concat3.conv.double_conv'))
        up2 = self.up_concat2(conv2, up3, params=self.get_subdict(params, 'up_concat2.conv.double_conv'))
        up1 = self.up_concat1(conv1, up2, params=self.get_subdict(params, 'up_concat1.conv.double_conv'))

        final = self.final(up1, params=self.get_subdict(params, 'final'))
        #final = F.log_softmax(final)
        final = torch.sigmoid(final)

        from data import visualize
        import matplotlib.pyplot as plt


        """visualize(inputs[0] , "input ")
        visualize(final.detach()[0], "output")
        mask = final.detach()[0] > 0.5
        visualize(mask, "mask")
        plt.show()"""

        """print("max and min value in output")
        print(torch.max(final))
        print(torch.min(final))"""

        return final




class unetConv2(MetaModule):
    def __init__(self, in_channels, out_channels, is_batchnorm, kernel_size=3, stride=1, padding=1, final=False):
        super(unetConv2, self).__init__()

        def init_layers(m):
            if type(m) == MetaConv2d:
                #torch.nn.init.xavier_uniform_(m.weight)
                n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels # or out_channels like in https://discuss.pytorch.org/t/unet-implementation/426/23
                torch.nn.init.normal_(m.weight, mean = 0.0, std = math.sqrt(2.0/n)) # from UNet Paper
                #torch.nn.init.kaiming_uniform_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        if is_batchnorm:
            
            self.double_conv = MetaSequential(OrderedDict([
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm1', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu', nn.ReLU()),
            ('conv2', MetaConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm2', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu', nn.ReLU())
            ]))

        else:
            self.double_conv = MetaSequential(OrderedDict([
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('relu', nn.ReLU()),
            ('conv2', MetaConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('relu', nn.ReLU())
            ]))

        # use the modules apply function to recursively apply the initialization
        self.double_conv.apply(init_layers)

    def forward(self, inputs, params=None):

        outputs = self.double_conv(inputs, params)
        return outputs


class unetUp(MetaModule):
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()

        self.conv = unetConv2(in_size, out_size, is_batchnorm=False)
        if is_deconv:
            self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2, params=None):

        outputs2 = self.up(inputs2)
        offset = outputs2.size()[2] - inputs1.size()[2]
        padding = 2 * [offset // 2, offset // 2]
        outputs1 = F.pad(inputs1, padding)
        return self.conv(torch.cat([outputs1, outputs2], 1), params)






