import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import time
from torch import nn as nn
from torch.nn import init as init
import torch.distributed as dist
from collections import OrderedDict





class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class CustomSequential(nn.Module):
    '''
    Similar to nn.Sequential, but it lets us introduce a second argument in the forward method
    so adaptors can be considered in the inference.
    '''

    def __init__(self, *args):
        super(CustomSequential, self).__init__()
        self.modules_list = nn.ModuleList(args)

    def forward(self, x, use_adapter=False):
        for module in self.modules_list:
            if hasattr(module, 'set_use_adapters'):
                module.set_use_adapters(use_adapter)
            x = module(x)
        return x





class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Adapter(nn.Module):

    def __init__(self, c, ffn_channel=None):
        super().__init__()
        if ffn_channel:
            ffn_channel = 2
        else:
            ffn_channel = c
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.depthwise = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1,
                                   groups=c, bias=True, dilation=1)

    def forward(self, input):

        x = self.conv1(input) + self.depthwise(input)
        x = self.conv2(x)

        return x


class FreMLP(nn.Module):

    def __init__(self, nc, expand=2):
        super(FreMLP, self).__init__()
        self.process1 = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)
        mag = self.process1(mag)
        real = mag * torch.cos(pha)
        imag = mag * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')
        return x_out


class Branch(nn.Module):
    '''
    Branch that lasts lonly the dilated convolutions
    '''

    def __init__(self, c, DW_Expand, dilation=1):
        super().__init__()
        self.dw_channel = DW_Expand * c

        self.branch = nn.Sequential(
            nn.Conv2d(in_channels=self.dw_channel, out_channels=self.dw_channel, kernel_size=3, padding=dilation,
                      stride=1, groups=self.dw_channel,
                      bias=True, dilation=dilation)  # the dconv
        )

    def forward(self, input):
        return self.branch(input)


class DBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        self.extra_conv = nn.Conv2d(self.dw_channel, self.dw_channel, kernel_size=3, padding=1, stride=1, groups=c,
                                    bias=True, dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(self.dw_channel, DW_Expand=1, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.sg2 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    #        self.adapter = Adapter(c, ffn_channel=None)

    #        self.use_adapters = False

    #    def set_use_adapters(self, use_adapters):
    #        self.use_adapters = use_adapters

    def forward(self, inp, adapter=None):

        y = inp
        x = self.norm1(inp)
        # x = self.conv1(self.extra_conv(x))
        x = self.extra_conv(self.conv1(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x = self.conv4(self.norm2(y))  # size [B, 2*C, H, W]
        x = self.sg2(x)  # size [B, C, H, W]
        x = self.conv5(x)  # size [B, C, H, W]
        x = y + x * self.gamma

        #        if self.use_adapters:
        #            return self.adapter(x)
        #        else:
        return x


class EBlock(nn.Module):
    '''
    Change this block using Branch
    '''

    def __init__(self, c, DW_Expand=2, dilations=[1], extra_depth_wise=False):
        super().__init__()
        # we define the 2 branches
        self.dw_channel = DW_Expand * c
        self.extra_conv = nn.Conv2d(c, c, kernel_size=3, padding=1, stride=1, groups=c, bias=True,
                                    dilation=1) if extra_depth_wise else nn.Identity()  # optional extra dw
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=self.dw_channel, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)

        self.branches = nn.ModuleList()
        for dilation in dilations:
            self.branches.append(Branch(c, DW_Expand, dilation=dilation))

        assert len(dilations) == len(self.branches)
        self.dw_channel = DW_Expand * c
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=self.dw_channel // 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True, dilation=1),
        )
        self.sg1 = SimpleGate()
        self.conv3 = nn.Conv2d(in_channels=self.dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True, dilation=1)
        # second step

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.freq = FreMLP(nc=c, expand=2)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    #        self.adapter = Adapter(c, ffn_channel=None)

    #        self.use_adapters = False

    #    def set_use_adapters(self, use_adapters):
    #        self.use_adapters = use_adapters

    def forward(self, inp):
        y = inp
        x = self.norm1(inp)
        x = self.conv1(self.extra_conv(x))
        z = 0
        for branch in self.branches:
            z += branch(x)

        z = self.sg1(z)
        x = self.sca(z) * z
        x = self.conv3(x)
        y = inp + self.beta * x
        # second step
        x_step2 = self.norm2(y)  # size [B, 2*C, H, W]
        x_freq = self.freq(x_step2)  # size [B, C, H, W]
        x = y * x_freq
        x = y + x * self.gamma

        #        if self.use_adapters:
        #            return self.adapter(x)
        #        else:
        return x

    # ----------------------------------------------------------------------------------------------


import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F

import cv2
import numpy as np


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        self.kernel_size = kernel_size[0]
        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        pad = self.kernel_size // 2
        input = F.pad(input, (pad, pad, pad, pad), mode='reflect')
        return self.conv(input, weight=self.weight, groups=self.groups)


class IllunimationConsistency(torch.nn.Module):
    def __init__(self, kernel_size=51, type='l2'):
        super(IllunimationConsistency, self).__init__()
        self.kernel_size = kernel_size
        # print('kernel_size',self.kernel_size)
        self.gaussian = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=0.1 * kernel_size)

    def down_smooth(self, x, scale=0.25):
        b, c, h, w = x.size()
        x_pack = x.contiguous().view(-1, 1, h, w)

        x_pack_blur = self.gaussian(x_pack)
        x_pack_blur = F.interpolate(x_pack_blur, scale_factor=scale, mode='bilinear', align_corners=False)
        return x_pack_blur

    def forward(self, x, y, scale=0.25):
        x_ds = self.down_smooth(x, scale)
        y_ds = self.down_smooth(y, scale)
        # print(x.shape,y.shape,x_ds.shape,y_ds.shape)
        return ((x_ds - y_ds) ** 2).mean()


class PyLap(torch.nn.Module):
    def __init__(self, num_levels=4):
        super(PyLap, self).__init__()
        self.num_levels = num_levels
        self.gaussian = GaussianSmoothing(1, 11, 1)

        nograd = [self.gaussian]
        for module in nograd:
            for param in module.parameters():
                param.requires_grad = False

    def _smooth(self, x):
        b, c, h, w = x.size()
        x_pack = x.view(-1, 1, h, w)

        x_pack_blur = self.gaussian(x_pack)
        return x_pack_blur.view(b, -1, h, w)

    def forward(self, data, inverse=False):
        if not inverse:
            pyramid = []
            current_level = data
            for level in range(self.num_levels):
                blurred = F.interpolate(current_level, scale_factor=0.5, mode='bilinear', align_corners=False)
                blurred = self._smooth(blurred)
                upsampled = F.interpolate(blurred, size=current_level.shape[2:], mode='bilinear', align_corners=False)
                residual = current_level - upsampled
                pyramid.append(residual)
                current_level = blurred

            pyramid.append(current_level)  # Add the lowest resolution image to the pyramid
            return pyramid
        else:
            restorer_x = data[-1]
            for level in range(len(data) - 2, -1, -1):
                # print(restorer_x.shape)

                restorer_x = F.interpolate(restorer_x, size=data[level].shape[2:], mode='bilinear', align_corners=False)

                # if level >0:
                restorer_x += data[level]
            return restorer_x


class MSRLayer(torch.nn.Module):
    def __init__(self, kernel_size):
        super(MSRLayer, self).__init__()
        self.kernel_size = kernel_size
        # print('kernel_size',self.kernel_size)
        self.gaussian = GaussianSmoothing(channels=1, kernel_size=kernel_size, sigma=0.1 * kernel_size)

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.masked_fill(x < 1e-3, 1e-3)

        x_pack = x.view(-1, 1, h, w)

        x_pack_blur = self.gaussian(x_pack)
        # print('x_pack',x_pack.shape,x_pack.max(),x_pack.mean(),x_pack.min())
        # print('x_pack_blur',x_pack_blur.shape,x_pack_blur.max(),x_pack_blur.mean(),x_pack_blur.min())
        dst_Img = torch.log(x_pack)
        # print('dst_Img ',dst_Img.shape,dst_Img.max(),dst_Img.mean(),dst_Img.min())

        dst_lblur = torch.log(x_pack_blur)
        # print('dst_lblur ',dst_lblur.shape,dst_lblur.max(),dst_lblur.mean(),dst_lblur.min())
        dst_Ixl = x_pack * x_pack_blur
        delta_i = dst_Img - dst_Ixl

        # input('cc')
        delta_i = delta_i.view(b * c, -1)

        outmap_min, _ = torch.min(delta_i, dim=1, keepdim=True)
        outmap_max, _ = torch.max(delta_i, dim=1, keepdim=True)
        # # print('outmap_min',outmap_min)
        delta_i = (delta_i - outmap_min) / (outmap_max - outmap_min)  # normalization

        return delta_i.view(b, c, h, w)


class GradientLoss(nn.Module):
    def __init__(self, weight=0.1):
        super(GradientLoss, self).__init__()
        self.weight = weight
        self.criterion = nn.L1Loss(reduction='mean')

    def cal_gradient(self, data):
        pad_y = torch.nn.functional.pad(data, (0, 0, 1, 0), mode='reflect')
        pad_x = torch.nn.functional.pad(data, (0, 1, 0, 0), mode='reflect')

        data_gy = pad_y[:, :, 1:] - pad_y[:, :, :-1]
        data_gx = pad_x[:, :, :, 1:] - pad_x[:, :, :, :-1]

        # print(data_gx.shape,data_gy.shape)
        return data_gx, data_gy

    def forward(self, pred, gt):
        # shape b 1 h w
        pred_gx, pred_gy = self.cal_gradient(pred)
        gt_gx, gt_gy = self.cal_gradient(gt)

        gx_loss = self.criterion(pred_gx, gt_gx) + self.criterion(pred_gy, gt_gy)
        return self.weight * gx_loss


class DarkIR(nn.Module):

    def __init__(self, img_channel=24,
                 width=32,
                 middle_blk_num_enc=2,
                 middle_blk_num_dec=2,
                 enc_blk_nums=[1, 2, 3],
                 dec_blk_nums=[3, 1, 1],
                 dilations=[1, 4, 9],
                 extra_depth_wise=True):
        super(DarkIR, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                CustomSequential(
                    *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks_enc = \
            CustomSequential(
                *[EBlock(chan, extra_depth_wise=extra_depth_wise) for _ in range(middle_blk_num_enc)]
            )
        self.middle_blks_dec = \
            CustomSequential(
                *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in
                  range(middle_blk_num_dec)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                CustomSequential(
                    *[DBlock(chan, dilations=dilations, extra_depth_wise=extra_depth_wise) for _ in range(num)]
                )
            )
        self.padder_size = 2 ** len(self.encoders)

        # this layer is needed for the computing of the middle loss. It isn't necessary for anything else
        self.side_out = nn.Conv2d(in_channels=width * 2 ** len(self.encoders), out_channels=img_channel,
                                  kernel_size=3, stride=1, padding=1)
        self.spy_lap = PyLap(3)
        self.window_size = 64

    def forward_step(self, input, side_loss=False, use_adapter=None):

        _, _, H, W = input.shape

        input = self.check_image_size(input)
        x = self.intro(input)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            skips.append(x)
            x = down(x)

        # we apply the encoder transforms
        x_light = self.middle_blks_enc(x)

        if side_loss:
            out_side = self.side_out(x_light)
        # apply the decoder transforms
        x = self.middle_blks_dec(x_light)
        x = x + x_light

        for decoder, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            x = up(x)
            x = x + skip
            x = decoder(x)

        x = self.ending(x)
        x = x + input
        out = x[:, :, :H, :W]  # we recover the original size of the image
        if side_loss:
            return out_side, out
        else:
            return out

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), value=0)
        return x

    def check_image_size_p(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img, inp_s1_img, gt=None):
        if gt is None:
            H, W = inp_img.shape[2:]
            inp_img = self.check_image_size_p(inp_img)
            inp_s1_img = self.check_image_size_p(inp_s1_img)

        # add lap decomposion
        b, c, h, w = inp_img.size()
        decomposition_ins = self.spy_lap(inp_img)
        decomposition_hn = self.spy_lap(inp_s1_img)

        high_frequency = []
        for i in range(len(decomposition_ins)):

            if i != 0:
                data_1 = F.interpolate(decomposition_ins[i], size=(h, w), mode='bilinear', align_corners=False)
                data_2 = F.interpolate(decomposition_hn[i], size=(h, w), mode='bilinear', align_corners=False)
            else:
                data_1 = decomposition_ins[i]
                data_2 = decomposition_hn[i]

            high_frequency += [data_1, data_2]

        base = decomposition_hn[-1]

        ins = torch.cat(high_frequency, 1)
        out_dec_level1 = self.forward_step(ins)
        spy_pred = []
        for i in range(len(decomposition_ins)):
            data = out_dec_level1[:, i * 3:(i + 1) * 3]
            # print(data.shape,decomposition_hn[i].shape)
            if i != 0:
                data = F.interpolate(data, size=(decomposition_hn[i].size()[-2], decomposition_hn[i].size()[-1]),
                                     mode='bilinear', align_corners=False)
                # if gt is None:
                # print(data.shape,decomposition_hn[i].shape)
                data = data + decomposition_hn[i]
            spy_pred.append(data)

        if gt is not None:
            spy_gt = self.spy_lap(gt)
            loss = 0
            for i in range(len(decomposition_ins)):
                loss += self.criterion(spy_pred[i], spy_gt[i])

            loss_ic = self.criterion(spy_pred[-1], decomposition_hn[-1])
            # + self.criterion(spy_pred[-2], decomposition_hn[-2]) + self.criterion(spy_pred[-3], decomposition_hn[-3])

            # pred = self.spy_lap(spy_pred,inverse=True)
            # loss_ic = self.fft_loss(spy_pred[-1], decomposition_hn[-1])*1e2

            loss_dict = dict(L1=loss, loss_ic=loss_ic)

            return loss_dict
        else:
            pred = self.spy_lap(spy_pred, inverse=True)
            inp_s1_img_re = self.spy_lap(decomposition_hn, inverse=True)
            return inp_s1_img_re[:, :, :H, :W], pred[:, :, :H, :W]


if __name__ == '__main__':
    img_channel = 24
    width = 32

    enc_blks = [1, 2, 3]
    middle_blk_num_enc = 2
    middle_blk_num_dec = 2
    dec_blks = [3, 1, 1]
    residual_layers = None
    dilations = [1, 4, 9]
    extra_depth_wise = True

    net = DarkIR(img_channel=img_channel,
                 width=width,
                 middle_blk_num_enc=middle_blk_num_enc,
                 middle_blk_num_dec=middle_blk_num_dec,
                 enc_blk_nums=enc_blks,
                 dec_blk_nums=dec_blks,
                 dilations=dilations,
                 extra_depth_wise=extra_depth_wise)



    net.eval()
    print('total parameters:', sum(param.numel() for param in net.parameters()) / 1e6)

    N = 2
    ins = torch.randn(1, 3, 64, 64)
    st = time.time()

    with torch.no_grad():
        for _ in range(N):
            _, output = net(ins, ins)
        print((time.time() - st) / N)
    print(output[-1].size())
    input('check')


