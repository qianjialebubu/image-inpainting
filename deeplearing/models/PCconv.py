from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torchvision.transforms as transforms
from .sdff import SDFF
import torch.nn.functional as F
import torch
import torch.nn as nn
import deeplearing.util.util as util
from deeplearing.util.Selfpatch import Selfpatch
from .ca import RAL
from .ASE import ase
#实验一：将sdff模块加入到特征融合中
# SE MODEL
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1)
        y = self.fc(y)
        return x * y.expand_as(x)


### 实验三 定义Attention门控块，在编码器到解码器的连接处增加注意力，使用Attention UNet
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        # 注意力门控向量
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        # 同层编码器特征图向量
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 卷积+BN+sigmoid激活函数
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1,
                      kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )


###  Attention门控的前向计算流程
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class Convnorm(nn.Module):
    def __init__(self, in_ch, out_ch, sample='none-3', activ='leaky'):
        super().__init__()
        self.bn = nn.InstanceNorm2d(out_ch, affine=True)

        if sample == 'down-3':
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 2, 1, bias=False)
        else:
            self.conv = nn.Conv2d(in_ch, out_ch, 3, 1)
        if activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input):
        out = input
        out = self.conv(out)
        out = self.bn(out)
        if hasattr(self, 'activation'):
            out = self.activation(out[0])
        return out


class PCBActiv(nn.Module):
    def __init__(self, in_ch, out_ch, bn=True, sample='none-3', activ='leaky',
                 conv_bias=False, innorm=False, inner=False, outer=False):
        super().__init__()
        if sample == 'same-5':
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == 'same-7':
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == 'down-3':
            self.conv = PartialConv(in_ch, out_ch, 3, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if bn:
            self.bn = nn.InstanceNorm2d(out_ch, affine=True)
        if activ == 'relu':
            self.activation = nn.ReLU()
        elif activ == 'leaky':
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.innorm = innorm
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])

        elif self.innorm:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            out[0] = self.activation(out[0])
        elif self.outer:
            out = self.conv(out)
            out[0] = self.bn(out[0])
        else:
            out = self.conv(out)
            out[0] = self.bn(out[0])
            if hasattr(self, 'activation'):
                out[0] = self.activation(out[0])
        return out


class ConvDown(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False, layers=1, activ=True):
        super().__init__()
        nf_mult = 1
        nums = out_c / 64
        sequence = []

        for i in range(1, layers + 1):
            nf_mult_prev = nf_mult
            if nums == 8:
                if in_c == 512:

                    nfmult = 1
                else:
                    nf_mult = 2

            else:
                nf_mult = min(2 ** i, 8)
            if kernel != 1:

                if activ == False and layers == 1:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c)
                    ]
                else:
                    sequence += [
                        nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                  kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                        nn.InstanceNorm2d(nf_mult * in_c),
                        nn.LeakyReLU(0.2, True)
                    ]

            else:

                sequence += [
                    nn.Conv2d(in_c, out_c,
                              kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                    nn.InstanceNorm2d(out_c),
                    nn.LeakyReLU(0.2, True)
                ]

            if activ == False:
                if i + 1 == layers:
                    if layers == 2:
                        sequence += [
                            nn.Conv2d(nf_mult * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    else:
                        sequence += [
                            nn.Conv2d(nf_mult_prev * in_c, nf_mult * in_c,
                                      kernel_size=kernel, stride=stride, padding=padding, bias=bias),
                            nn.InstanceNorm2d(nf_mult * in_c)
                        ]
                    break

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


class ConvUp(nn.Module):
    def __init__(self, in_c, out_c, kernel, stride, padding=0, dilation=1, groups=1, bias=False):
        super().__init__()

        self.conv = nn.Conv2d(in_c, out_c, kernel,
                              stride, padding, dilation, groups, bias)
        self.bn = nn.InstanceNorm2d(out_c)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode='bilinear')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


# 递归门控卷积
def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)
class gnconv(nn.Module):
    def __init__(self, dim, order=3, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order# 空间交互的阶数，即n
        self.dims = [dim // 2 ** i for i in range(order)]# 将2C在不同阶的空间上进行切分，对应公式3.2
        self.dims.reverse()# 反序，使低层通道少，高层通道多
        self.proj_in = nn.Conv2d(dim, 2* dim, 1)# 输入x的线性映射层，对应$\phi(in)$
        if gflayer is None:# 是否使用Global Filter
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:# 在全特征上进行卷积，多在后期使用
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.proj_out = nn.Conv2d(dim, dim, 1)# 输出y的线性映射层，对应$\phi(out)$
        self.pws = nn.ModuleList(# 高阶空间交互过程中使用的卷积模块，对应公式3.4
            [nn.Conv2d(self.dims[i], self.dims[i + 1], 1) for i in range(order - 1)]
        )
        self.scale = s# 缩放系数，对应公式3.3中的$\alpha$
    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in(x)# channel double
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)# split channel to c/2**order and c(1-2**oder)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order - 1):
            x = self.pws[i](x) * dw_list[i + 1]
        x = self.proj_out(x)
        return x

class BASE(nn.Module):
    def __init__(self, inner_nc):
        super(BASE, self).__init__()
        se = SELayer(inner_nc, 16)
        model = [se]
        gus = util.gussin(1.5).cuda()
        self.gus = torch.unsqueeze(gus, 1).double()
        self.model = nn.Sequential(*model)
        self.down = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, 0, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2)
        )
        self.fuse = ConvDown(512 * 2, 512, 1, 1)
        self.ral = RAL(kernel_size=3, stride=1, rate=2, softmax_scale=10.)
        self.gnconv = gnconv(inner_nc)

    def forward(self, x):
        Nonparm = Selfpatch()
        out_32 = self.model(x)

        out_res = self.gnconv(out_32)
        out_32 = self.ral(out_32, out_32)
        out_32 = self.gnconv(out_32)
        b, c, h, w = out_32.size()
        gus = self.gus.float()
        gus_out = out_32[0].expand(h * w, c, h, w)
        gus_out = gus * gus_out
        gus_out = torch.sum(gus_out, -1)
        gus_out = torch.sum(gus_out, -1)
        gus_out = gus_out.contiguous().view(b, c, h, w)
        csa2_in = F.sigmoid(out_32)
        csa2_f = torch.nn.functional.pad(csa2_in, (1, 1, 1, 1))
        csa2_ff = torch.nn.functional.pad(out_32, (1, 1, 1, 1))
        csa2_fff, csa2_f, csa2_conv = Nonparm.buildAutoencoder(csa2_f[0], csa2_in[0], csa2_ff[0], 3, 1)
        csa2_conv = csa2_conv.expand_as(csa2_f)
        csa_a = csa2_conv * csa2_f
        csa_a = torch.mean(csa_a, 1)
        a_c, a_h, a_w = csa_a.size()
        csa_a = csa_a.contiguous().view(a_c, -1)
        csa_a = F.softmax(csa_a, dim=1)
        csa_a = csa_a.contiguous().view(a_c, 1, a_h, a_h)
        out = csa_a * csa2_fff
        out = torch.sum(out, -1)
        out = torch.sum(out, -1)
        out_csa = out.contiguous().view(b, c, h, w)
        out_32 = torch.cat([gus_out, out_csa], 1)
        out_32 = self.down(out_32)

        out_32 = torch.cat([out_32, out_res], 1)
        out_32 = self.fuse(out_32)

        return out_32


class PartialConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias)
        self.mask_conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                                   stride, padding, dilation, groups, False)

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, inputt):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)

        input = inputt[0]
        mask = inputt[1].float().cuda()

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(
                output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = []
        out.append(output)
        out.append(new_mask)
        return out


class PCconv(nn.Module):
    def __init__(self):
        super(PCconv, self).__init__()
        self.down_128 = ConvDown(64, 128, 4, 2, padding=1, layers=2)
        self.down_64 = ConvDown(128, 256, 4, 2, padding=1)
        self.down_32 = ConvDown(256, 256, 1, 1)
        self.down_16 = ConvDown(512, 512, 4, 2, padding=1, activ=False)
        self.down_8 = ConvDown(512, 512, 4, 2, padding=1, layers=2, activ=False)
        self.down_4 = ConvDown(512, 512, 4, 2, padding=1, layers=3, activ=False)
        self.down = ConvDown(768, 256, 1, 1)
        self.fuse = ConvDown(512, 512, 1, 1)
        self.up = ConvUp(512, 256, 1, 1)
        self.up_128 = ConvUp(512, 64, 1, 1)
        self.up_64 = ConvUp(512, 128, 1, 1)
        self.up_32 = ConvUp(512, 256, 1, 1)
        self.base= BASE(512)
        seuqence_3 = []
        seuqence_5 = []
        seuqence_7 = []
        for i in range(5):
            seuqence_3 += [PCBActiv(256, 256, innorm=True)]
            seuqence_5 += [PCBActiv(256, 256, sample='same-5', innorm=True)]
            seuqence_7 += [PCBActiv(256, 256, sample='same-7', innorm=True)]

        self.cov_3 = nn.Sequential(*seuqence_3)
        self.cov_5 = nn.Sequential(*seuqence_5)
        self.cov_7 = nn.Sequential(*seuqence_7)
        self.activation = nn.LeakyReLU(negative_slope=0.2)
        self.attUnet6 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.attUnet5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.attUnet4 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.attUnet3 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.attUnet2 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.attUnet1 = Attention_block(F_g=64,F_l=64,F_int=32)

        self.detail_structure_fusion = SDFF(256, 256)
    def forward(self, input, mask):
        mask =  util.cal_feat_mask(mask, 3, 1)
        # input[2]:256 32 32
        # mask:[1,1,32,32]
        b, c, h, w = input[2].size()
        mask_1 = torch.add(torch.neg(mask.float()), 1)
        mask_1 = mask_1.expand(b, c, h, w)
        # mapping_image(mask_1,256,'mask_1')
        x_1 = self.activation(input[0])
        x_2 = self.activation(input[1])
        x_3 = self.activation(input[2])
        x_4 = self.activation(input[3])
        x_5 = self.activation(input[4])
        x_6 = self.activation(input[5])
        # Change the shape of each layer and intergrate low-level/high-level features
        x_1 = self.down_128(x_1)
        x_2 = self.down_64(x_2)
        x_3 = self.down_32(x_3)
        x_4 = self.up(x_4, (32, 32))
        x_5 = self.up(x_5, (32, 32))
        x_6 = self.up(x_6, (32, 32))

        # The first three layers are Texture/detail
        # The last three layers are Structure
        x_DE = torch.cat([x_1, x_2, x_3], 1)
        mapping_image(x_DE, 768, 'x_DE_768')
        x_ST = torch.cat([x_4, x_5, x_6], 1)
        mapping_image(x_ST, 768, 'x_ST_768')
        x_ST = self.down(x_ST)
        mapping_image(x_ST, 256, 'x_ST_256')
        x_DE = self.down(x_DE)
        mapping_image(x_DE, 256, 'x_DE_256')
        x_ST = [x_ST, mask_1]
        x_DE = [x_DE, mask_1]

        # Multi Scale PConv fill the Details
        x_DE_3 = self.cov_3(x_DE)
        mapping_image(x_DE_3[0],256,'x_DE_3[0]')
        x_DE_5 = self.cov_5(x_DE)
        x_DE_7 = self.cov_7(x_DE)
        x_DE_fuse = torch.cat([x_DE_3[0], x_DE_5[0], x_DE_7[0]], 1)
        mapping_image(x_DE_fuse, 768, 'x_DE_fuse')
        x_DE_fi = self.down(x_DE_fuse)
        mapping_image(x_DE_fi, 256, 'x_DE_fi')
        # Multi Scale PConv fill the Structure
        x_ST_3 = self.cov_3(x_ST)
        x_ST_5 = self.cov_5(x_ST)
        x_ST_7 = self.cov_7(x_ST)
        x_ST_fuse = torch.cat([x_ST_3[0], x_ST_5[0], x_ST_7[0]], 1)
        mapping_image(x_ST_fuse, 768, 'x_ST_fuse')
        x_ST_fi = self.down(x_ST_fuse)
        mapping_image(x_ST_fi, 256, 'x_ST_fi')
        # 这里是直接使用cat连接，sdff是使用sdff模块进行连接
        # 实验一
        # x_cat = torch.cat([x_ST_fi, x_DE_fi], 1)
        x_cat = self.detail_structure_fusion(x_ST_fi, x_DE_fi)
        mapping_image(x_cat, 512, 'x_cat')
        x_cat_fuse = self.fuse(x_cat)
        mapping_image(x_cat_fuse, 512, 'x_cat_fuse')
        # Feature equalizations
        x_final = self.base(x_cat_fuse)
        mapping_image(x_final, 512, 'x_final')
        # Add back to the input
        x_ST = x_final
        x_DE = x_final
        # x_1 = self.up_128(x_DE, (128, 128)) + input[0]
        x_1 = self.attUnet1(self.up_128(x_DE,(128,128)),input[0])
        # x_1 = ase(x_1)
        # x_2 = self.up_64(x_DE, (64, 64)) + input[1]
        x_2 = self.attUnet2(self.up_64(x_DE, (64, 64)) , input[1])
        # x_2 = ase(x_2)
        # x_3 = self.up_32(x_DE, (32, 32)) + input[2]
        x_3 = self.attUnet3(self.up_32(x_DE, (32, 32)) , input[2])
        # x_3 = ase(x_3)
        # x_4 = self.down_16(x_ST) + input[3]
        x_4 = self.attUnet4(self.down_16(x_ST) , input[3])
        # x_5 = self.down_8(x_ST) + input[4]
        x_5 = self.attUnet5(self.down_8(x_ST) , input[4])
        # x_6 = self.down_4(x_ST) + input[5]
        x_6 = self.attUnet6(self.down_4(x_ST) , input[5])

        out = [x_1, x_2, x_3, x_4, x_5, x_6]
        loss = [x_ST_fi, x_DE_fi]
        out_final = [out, loss]
        return out_final


def mapping_image(input,input_channels,save_name):
    x_cattt = input.cpu()

    con = nn.Conv2d(input_channels, 3, kernel_size=1, stride=1, padding=0)
    x_catt = con(x_cattt)
    x_catt = F.interpolate(x_catt, size=(256, 256), mode='bilinear')
    feature_map_cpu = x_catt.cpu()
    # 转换为范围为 [0, 1] 的张量
    feature_map = (feature_map_cpu - feature_map_cpu.min()) / (feature_map_cpu.max() - feature_map_cpu.min())
    # 转换为 PIL 图像对象
    transform = transforms.ToPILImage()
    image = transform(feature_map[0])
    # 保存为图像文件
    image.save(save_name + '.jpg')