import torch
import torch.nn as nn
import torch.nn.functional as F

from torchmeta.modules import (MetaModule, MetaConv2d, MetaSequential)
from collections import OrderedDict

import math
from mymodules import MetaConvTranspose2d


"""--------------------------------------- simple UNet implementation --------------------------------------------------"""


class Unet(MetaModule):
    def __init__(
        self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, padding=1, device='cpu'
    ):
        super(Unet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, device=device, padding=padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, device=device, padding=padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, device=device, padding=padding)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm,  device=device, padding=padding)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.center = unetConv2(filters[3], filters[4], self.is_batchnorm, device=device, padding=padding)

        # Upsampling
        self.up_concat4 = unetUp(filters[4], filters[3], self.is_deconv)
        self.up_concat3 = unetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = unetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = unetUp(filters[1], filters[0], self.is_deconv)




        # Final convolution layer (without any concat)
        self.final = MetaConv2d(filters[0], n_classes, kernel_size=1)


    def forward(self, inputs, params=None):

        conv1 = self.conv1(inputs, params=self.get_subdict(params, 'conv1.double_conv'))
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1, params=self.get_subdict(params, 'conv2.double_conv'))
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2, params=self.get_subdict(params, 'conv3.double_conv'))
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3, params=self.get_subdict(params, 'conv4.double_conv'))
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4, params=self.get_subdict(params, 'center.double_conv'))

        up4 = self.up_concat4(conv4, center, params_conv=self.get_subdict(params, 'up_concat4.conv.double_conv'), params_up=self.get_subdict(params, 'up_concat4.up'))
        up3 = self.up_concat3(conv3, up4, params_conv=self.get_subdict(params, 'up_concat3.conv.double_conv'), params_up=self.get_subdict(params, 'up_concat3.up'))
        up2 = self.up_concat2(conv2, up3, params_conv=self.get_subdict(params, 'up_concat2.conv.double_conv'), params_up=self.get_subdict(params, 'up_concat2.up'))
        up1 = self.up_concat1(conv1, up2, params_conv=self.get_subdict(params, 'up_concat1.conv.double_conv'), params_up=self.get_subdict(params, 'up_concat1.up'))

        final = self.final(up1, params=self.get_subdict(params, 'final'))

        return final




class unetConv2(MetaModule):
    """Double convolution modules: (conv + norm + relu) x2"""
    def __init__(self, in_channels, out_channels, is_batchnorm, kernel_size=3, stride=1, padding=1, final=False, device='cpu', fcn=False, fcn_center=False):
        super(unetConv2, self).__init__()
        self.device=device

        def init_layers(m):
            if type(m) == MetaConv2d:
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')

        if fcn==True:
            self.double_conv = MetaSequential(OrderedDict([
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm1', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu1', nn.ReLU(inplace=True)),
            #('dropout1', nn.Dropout(0.3)),
            ('conv2', MetaConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm2', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu2', nn.ReLU(inplace=True)),
            #('dropout2', nn.Dropout(0.3))
            ('conv3', MetaConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm3', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu3', nn.ReLU(inplace=True))
            ]))
        elif fcn_center==True:
            self.double_conv = MetaSequential(OrderedDict([
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm1', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu1', nn.ReLU(inplace=True))
            ]))


        else:
            
            self.double_conv = MetaSequential(OrderedDict([
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm1', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu1', nn.ReLU(inplace=True)),
            #('dropout1', nn.Dropout(0.3)),
            ('conv2', MetaConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm2', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu2', nn.ReLU(inplace=True))
            #('dropout2', nn.Dropout(0.3))
            ]))

        # Use the module's apply function to recursively apply the initialization
        self.double_conv.apply(init_layers)

    def forward(self, inputs, params=None):
        if self.device=='cuda':
            inputs = inputs.to(self.device)
        outputs = self.double_conv(inputs, params)
        return outputs


class unetUp(MetaModule):
    """Upsampling modules"""
    def __init__(self, in_size, out_size, is_deconv):
        super(unetUp, self).__init__()

        self.conv = unetConv2(in_size, out_size, is_batchnorm=True)
        if is_deconv:
            #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            self.up = MetaConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2, params_conv=None, params_up=None):
        outputs2 = self.up(inputs2, params=params_up)
        #offset = outputs2.size()[2] - inputs1.size()[2]
        #padding = 2 * [offset // 2, offset // 2]
        #outputs1 = F.pad(inputs1, padding)
        outputs1 = inputs1
        return self.conv(torch.cat([outputs1, outputs2], 1), params=params_conv)



"""--------------------------------------- ResUnet implementation --------------------------------------------------"""


class ResUnet(MetaModule):
    def __init__(
        self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, padding=1, device='cpu'
    ):
        super(ResUnet, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling
        
        self.conv1 = ResUnetConv2(self.in_channels, filters[0], device=device, padding=padding, is_first=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.conv2 = ResUnetConv2(filters[0], filters[1], device=device, padding=padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.conv3 = ResUnetConv2(filters[1], filters[2], device=device, padding=padding)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
        
        self.center = ResUnetConv2(filters[2], filters[3], device=device, padding=padding)

        # Upsampling
        
        self.up_concat3 = ResUnetUp(filters[3], filters[2], self.is_deconv)
        self.up_concat2 = ResUnetUp(filters[2], filters[1], self.is_deconv)
        self.up_concat1 = ResUnetUp(filters[1], filters[0], self.is_deconv)


        # Final convolution layer (without any concat)
        self.final = MetaConv2d(filters[0], n_classes, kernel_size=1)


    def forward(self, inputs, params=None):
        conv1 = self.conv1(inputs, params_conv=self.get_subdict(params, 'conv1.double_conv'), params_add=self.get_subdict(params, 'conv1.addition_connection'))
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1, params_conv=self.get_subdict(params, 'conv2.double_conv'), params_add=self.get_subdict(params, 'conv2.addition_connection'))
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2, params_conv=self.get_subdict(params, 'conv3.double_conv'), params_add=self.get_subdict(params, 'conv3.addition_connection'))
        maxpool3 = self.maxpool3(conv3)

        center = self.center(maxpool3, params_conv=self.get_subdict(params, 'center.double_conv'), params_add=self.get_subdict(params, 'center.addition_connection'))

        #final = torch.sigmoid(final)
        up3 = self.up_concat3(conv3, center, params_conv=self.get_subdict(params, 'up_concat3.conv.double_conv'), params_up=self.get_subdict(params, 'up_concat3.up'), params_add=self.get_subdict(params, 'up_concat3.conv.addition_connection'))
        up2 = self.up_concat2(conv2, up3, params_conv=self.get_subdict(params, 'up_concat2.conv.double_conv'), params_up=self.get_subdict(params, 'up_concat2.up'), params_add=self.get_subdict(params, 'up_concat2.conv.addition_connection'))
        up1 = self.up_concat1(conv1, up2, params_conv=self.get_subdict(params, 'up_concat1.conv.double_conv'), params_up=self.get_subdict(params, 'up_concat1.up'), params_add=self.get_subdict(params, 'up_concat1.conv.addition_connection'))

        final = self.final(up1, params=self.get_subdict(params, 'final'))

        return final




class ResUnetConv2(MetaModule):
    """Double convolution modules: (conv + norm + relu) x2"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, final=False, device='cpu', is_first=False):
        super(ResUnetConv2, self).__init__()
        self.device=device

        def init_layers(m):
            if type(m) == MetaConv2d:
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


        if is_first:

            self.double_conv = MetaSequential(OrderedDict([
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm1', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu1', nn.ReLU(inplace=True)),
            #('dropout1', nn.Dropout(0.3)),
            ('conv2', MetaConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            #('dropout2', nn.Dropout(0.3))
            ]))

        else:
            
            self.double_conv = MetaSequential(OrderedDict([
            ('norm1', nn.BatchNorm2d(in_channels)),# momentum=1.,track_running_stats=False)),
            ('relu1', nn.ReLU(inplace=True)),
            #('dropout1', nn.Dropout(0.3)),
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm2', nn.BatchNorm2d(out_channels)),# momentum=1.,track_running_stats=False)),
            ('relu2', nn.ReLU(inplace=True)),
            ('conv2', MetaConv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding))
            #('dropout2', nn.Dropout(0.3))
            ]))

        self.addition_connection = MetaSequential(OrderedDict([
            ('conv1', MetaConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)),
            ('norm2', nn.BatchNorm2d(out_channels))# momentum=1.,track_running_stats=False)),
            #('dropout2', nn.Dropout(0.3))
            ]))

        # use the module's apply function to recursively apply the initialization
        self.double_conv.apply(init_layers)

    def forward(self, inputs, params_conv=None, params_add=None):
        if self.device=='cuda':
            inputs = inputs.to(self.device)
        return self.double_conv(inputs, params=params_conv) + self.addition_connection(inputs, params=params_add)


class ResUnetUp(MetaModule):
    """Upsampling modules"""
    def __init__(self, in_size, out_size, is_deconv):
        super(ResUnetUp, self).__init__()

        self.conv = ResUnetConv2(in_size, out_size)

        if is_deconv:
            #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            self.up = MetaConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)


    def forward(self, inputs1, inputs2,  params_conv=None, params_up=None, params_add=None):
        outputs2 = self.up(inputs2, params=params_up)
        
        #offset = outputs2.size()[2] - inputs1.size()[2]
        #padding = 2 * [offset // 2, offset // 2]
        #outputs1 = F.pad(inputs1, padding)
        outputs1 = inputs1
        concat = torch.cat([outputs1, outputs2], 1)
        block_output = self.conv(concat, params_conv=params_conv, params_add=params_add)
        return block_output

        



"""--------------------------------------------- FCN8 -----------------------------------__"""


class FCN8(MetaModule):
    def __init__(
        self, feature_scale=4, n_classes=1, is_deconv=True, in_channels=3, is_batchnorm=True, padding=1, device='cpu'
    ):
        super(FCN8, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_batchnorm = is_batchnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024, 2048]
        filters = [int(x / self.feature_scale) for x in filters]

        # Downsampling
        self.conv1 = unetConv2(self.in_channels, filters[0], self.is_batchnorm, device=device, padding=padding)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv2 = unetConv2(filters[0], filters[1], self.is_batchnorm, device=device, padding=padding)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv3 = unetConv2(filters[1], filters[2], self.is_batchnorm, device=device, padding=padding, fcn=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv4 = unetConv2(filters[2], filters[3], self.is_batchnorm,  device=device, padding=padding, fcn=True)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.conv5 = unetConv2(filters[3], filters[4], self.is_batchnorm,  device=device, padding=padding, fcn=True)
        self.maxpool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)


        self.center1 = unetConv2(filters[4], filters[4], self.is_batchnorm,  device=device, padding=padding, fcn_center=True)
        self.center2 = unetConv2(filters[4], n_classes, self.is_batchnorm,  device=device, padding=padding, fcn_center=True)

        self.up_center = fcnUp(n_classes, n_classes, self.is_deconv)

        self.down4 = unetConv2(filters[3], n_classes, self.is_batchnorm,  device=device, padding=padding, fcn_center=True)
        self.maxpool4down = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.down3 = unetConv2(filters[2], n_classes, self.is_batchnorm,  device=device, padding=padding, fcn_center=True)
        self.maxpool3down = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

        self.add1up = fcnUp(n_classes, n_classes, self.is_deconv)

        self.up = fcnUp(n_classes, n_classes, self.is_deconv)
        self.up2 = fcnUp(n_classes, n_classes, self.is_deconv)
        self.up3 = fcnUp(n_classes, n_classes, self.is_deconv)
        



    def forward(self, inputs, params=None):
        conv1 = self.conv1(inputs, params=self.get_subdict(params, 'conv1.double_conv'))
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1, params=self.get_subdict(params, 'conv2.double_conv'))
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2, params=self.get_subdict(params, 'conv3.double_conv'))
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3, params=self.get_subdict(params, 'conv4.double_conv'))
        maxpool4 = self.maxpool4(conv4)

        conv5 = self.conv5(maxpool4, params=self.get_subdict(params, 'conv5.double_conv'))
        maxpool5 = self.maxpool5(conv5)

        center1 = self.center1(maxpool5, params=self.get_subdict(params, 'center1.double_conv'))
        center2 = self.center2(center1, params=self.get_subdict(params, 'center2.double_conv'))

        up_center = self.up_center(center2, inputs2=None, params=self.get_subdict(params, 'up_center.up'))

        down4 = self.down4(conv4, params=self.get_subdict(params, 'down4.double_conv'))
        down4 = self.maxpool4down(down4)

        add1 = down4+up_center

        down3 = self.down3(conv3, params=self.get_subdict(params, 'down3.double_conv'))
        down3 = self.maxpool3down(down3)


        add1_up = self.add1up(add1, None, params=self.get_subdict(params, 'add1up.up'))
        add2 = add1_up+down3

        up = self.up(add2, None, params=self.get_subdict(params, 'up.up'))
        up2 = self.up2(up, None, params=self.get_subdict(params, 'up2.up'))
        up3 = self.up3(up2, None, params=self.get_subdict(params, 'up3.up'))
    
        return up3




class fcnUp(MetaModule):
    """Upsampling modules"""
    def __init__(self, in_size, out_size, is_deconv):
        super(fcnUp, self).__init__()
 
        if is_deconv:
            #self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
            self.up = MetaConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
        else:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, inputs1, inputs2, params=None):
        if inputs2==None:
            inputs = self.up(inputs1, params=params)
            return inputs
