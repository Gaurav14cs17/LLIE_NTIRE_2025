from torch.nn.init import _calculate_fan_in_and_fan_out
from torch.nn.init import trunc_normal_
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
from einops import rearrange
import sys
import math
import torch
from torch import nn
from torch.nn import init
from torch.nn import functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# Define IGAB (Illumination-Guided Attention Block)
import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as stx
import numbers
import warnings
from einops import rearrange
import math

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



class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class KANLinear(torch.nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            grid_size=5,
            spline_order=3,
            scale_noise=0.1,
            scale_base=1.0,
            scale_spline=1.0,
            enable_standalone_scale_spline=True,
            base_activation=torch.nn.SiLU,
            grid_eps=0.02,
            grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                    torch.arange(-spline_order, grid_size + spline_order + 1) * h
                    + grid_range[0]
            )
                .expand(in_features, -1)
                .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                    (
                            torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                            - 1 / 2
                    )
                    * self.scale_noise
                    / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                            (x - grid[:, : -(k + 1)])
                            / (grid[:, k:-1] - grid[:, : -(k + 1)])
                            * bases[:, :, :-1]
                    ) + (
                            (grid[:, k + 1:] - x)
                            / (grid[:, k + 1:] - grid[:, 1:(-k)])
                            * bases[:, :, 1:]
                    )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                torch.arange(
                    self.grid_size + 1, dtype=torch.float32, device=x.device
                ).unsqueeze(1)
                * uniform_step
                + x_sorted[0]
                - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
                regularize_activation * regularization_loss_activation
                + regularize_entropy * regularization_loss_entropy
        )


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.bn = nn.BatchNorm2d(dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class kan(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5,
                 version=4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        self.fc1 = KANLinear(
            in_features,
            hidden_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.fc3 = KANLinear(
            hidden_features,
            out_features,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)

        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)

        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x








def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def variance_scaling_(tensor, scale=1.0, mode='fan_in', distribution='normal'):
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    if mode == 'fan_in':
        denom = fan_in
    elif mode == 'fan_out':
        denom = fan_out
    elif mode == 'fan_avg':
        denom = (fan_in + fan_out) / 2
    variance = scale / denom
    if distribution == "truncated_normal":
        trunc_normal_(tensor, std=math.sqrt(variance) / .87962566103423978)
    elif distribution == "normal":
        tensor.normal_(std=math.sqrt(variance))
    elif distribution == "uniform":
        bound = math.sqrt(3 * variance)
        tensor.uniform_(-bound, bound)
    else:
        raise ValueError(f"invalid distribution {distribution}")


def lecun_normal_(tensor):
    variance_scaling_(tensor, mode='fan_in', distribution='truncated_normal')


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)


# input [bs,28,256,310]  output [bs, 28, 256, 256]
def shift_back(inputs, step=2):
    [bs, nC, row, col] = inputs.shape
    down_sample = 256 // row
    step = float(step) / float(down_sample * down_sample)
    out_col = row
    for i in range(nC):
        inputs[:, i, :, :out_col] = \
            inputs[:, i, :, int(step * i):int(step * i) + out_col]
    return inputs[:, :, :, :out_col]


class Illumination_Estimator(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=24, n_fea_out=3):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(Illumination_Estimator, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)

        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=4)

        # self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        # mean_c = img.mean(dim=1).unsqueeze(1)
        # stx()
        # input = torch.cat([img,mean_c], dim=1)

        x_1 = self.conv1(img)
        illu_fea = self.depth_conv(x_1)
        # illu_map = self.conv2(illu_fea)
        return illu_fea


class IG_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(v_inp.reshape(b, h, w, c).permute(
            0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class IGAB(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                IG_MSA(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)
        return out


class Denoiser(nn.Module):
    def __init__(self, in_dim=20, out_dim=3, dim=31, level=2, num_blocks=[2, 4, 4]):
        super(Denoiser, self).__init__()
        self.dim = dim
        self.level = level

        # Input projection
        self.embedding = nn.Conv2d(24, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                IGAB(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = IGAB(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                KANLinear(in_features=dim_level // 2, out_features=dim_level // 2) , # KAN layer in skip connection
                IGAB(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        self.mapping = nn.Conv2d(self.dim, 12, 3, 1, 1, bias=False)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

        self.criterion = nn.L1Loss(reduction='mean')
        self.spy_lap = PyLap(4)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding(x)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        for (IGAB, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea = IGAB(fea, illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea = self.bottleneck(fea, illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, KANLayer, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            skip_connection = fea_encoder[self.level - 1 - i]
            skip_connection = KANLayer(skip_connection.permute(0, 2, 3, 1).reshape(-1, skip_connection.size(1))).reshape(skip_connection.size())
            fea = Fution(torch.cat([fea, skip_connection], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea = LeWinBlcok(fea, illu_fea)

        # Mapping
        out = self.mapping(fea)

        return out


class RetinexFormer_Single_Stage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=2, num_blocks=[1, 1, 1]):
        super(RetinexFormer_Single_Stage, self).__init__()
        self.estimator = Illumination_Estimator(n_feat)
        self.denoiser = Denoiser(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                                 num_blocks=num_blocks)  #### 将 Denoiser 改为 img2img

    def forward(self, ins1_decompose, ins2_decompose=None):
        h, w = ins1_decompose[0].size()[-2:]
        if ins2_decompose is None:
            high_frequency = []
            for i in range(len(ins1_decompose)):
                if i != 0:
                    data_1 = F.interpolate(ins1_decompose[i], size=(h, w), mode='bilinear', align_corners=False)
                    data_2 = F.interpolate(ins1_decompose[i], size=(h, w), mode='bilinear', align_corners=False)
                else:
                    data_1 = ins1_decompose[i]
                    data_2 = ins1_decompose[i]
                high_frequency += [data_1, data_2]

        else:
            high_frequency = []
            for i in range(len(ins1_decompose)):
                if i != 0:
                    data_1 = F.interpolate(ins1_decompose[i], size=(h, w), mode='bilinear', align_corners=False)
                    data_2 = F.interpolate(ins2_decompose[i], size=(h, w), mode='bilinear', align_corners=False)
                else:
                    data_1 = ins1_decompose[i]
                    data_2 = ins2_decompose[i]
                high_frequency += [data_1, data_2]

        ins = torch.cat(high_frequency, 1)

        illu_fea = self.estimator(ins)
        # input_img = inp_img * illu_map + inp_img
        out_dec_level1 = self.denoiser(ins, illu_fea)

        spy_pred = []
        for i in range(len(ins1_decompose)):
            data = out_dec_level1[:, i * 3:(i + 1) * 3]
            # print(data.shape,decomposition_hn[i].shape)
            if i != 0:
                data = F.interpolate(data, size=(ins2_decompose[i].size()[-2], ins2_decompose[i].size()[-1]),
                                     mode='bilinear', align_corners=False)
                # if gt is None:
                # print(data.shape,decomposition_hn[i].shape)
                data = data + ins2_decompose[i]
            spy_pred.append(data)

        return spy_pred


class RetinexFormer(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, stage=3, num_blocks=[1, 1, 1]):
        super(RetinexFormer, self).__init__()
        self.stage = stage

        modules_body = [
            RetinexFormer_Single_Stage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=2,
                                       num_blocks=num_blocks)
            for _ in range(stage)]

        self.body = nn.Sequential(*modules_body)

        self.criterion = nn.L1Loss(reduction='mean')

        self.spy_lap = PyLap(3)

        self.window_size = 64

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, inp_img, inp_s1_img, gt=None):

        if gt is None:
            H, W = inp_img.shape[2:]
            inp_img = self.check_image_size(inp_img)
            inp_s1_img = self.check_image_size(inp_s1_img)
        b, c, h, w = inp_img.size()
        decomposition_ins = self.spy_lap(inp_img)
        decomposition_hn = self.spy_lap(inp_s1_img)

        decomposition_update = decomposition_ins
        for block in self.body:
            decomposition_update = block(decomposition_update, decomposition_hn)

        if gt is not None:
            spy_gt = self.spy_lap(gt)
            loss = 0
            for i in range(len(decomposition_update)):
                loss += self.criterion(decomposition_update[i], spy_gt[i])

            loss_ic = self.criterion(decomposition_update[-1], decomposition_hn[-1])

            loss_dict = dict(L1=loss, loss_ic=loss_ic)

            return loss_dict
        else:
            pred = self.spy_lap(decomposition_update, inverse=True)
            inp_s1_img_re = self.spy_lap(decomposition_hn, inverse=True)
            return inp_s1_img_re[:, :, :H, :W], pred[:, :, :H, :W]


if __name__ == "__main__":
    import time

    # from fvcore.nn import FlopCountAnalysis
    net = RetinexFormer(stage=1, n_feat=40, num_blocks=[1, 2, 2])

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