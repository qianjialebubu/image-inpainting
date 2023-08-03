import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
class ECABlock(nn.Module):
    def __init__(self, channels, gamma = 2, b = 1):
        super(ECABlock, self).__init__()

        # 设计自适应卷积核，便于后续做1*1卷积
        kernel_size = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1

        # 全局平局池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 基于1*1卷积学习通道之间的信息
        self.conv = nn.Conv1d(1, 1, kernel_size = kernel_size, padding = (kernel_size - 1) // 2, bias = False)

        # 激活函数
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 首先，空间维度做全局平局池化，[b,c,h,w]==>[b,c,1,1]
        v = self.avg_pool(x)

        # 然后，基于1*1卷积学习通道之间的信息；其中，使用前面设计的自适应卷积核
        v = self.conv(v.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # 最终，经过sigmoid 激活函数处理
        v = self.sigmoid(v)
        return x * v




# v0.1 sdff模块的改造，使用结构和纹理一致的方法进行实验,增加了一个可学习的参数
# SK MODEL
#
# 把sdff的senet改为sknet
class SKConv(nn.Module):
    def __init__(self, features, WH, M, G, r, stride=1, L=32):
        super(SKConv, self).__init__()
        d = max(int(features / r), L)
        self.M = M
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(M):
            # 使用不同kernel size的卷积
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(features,
                              features,
                              kernel_size=3 + i * 2,
                              stride=stride,
                              padding=1 + i,
                              groups=G), nn.BatchNorm2d(features),
                    nn.ReLU(inplace=False)))

        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(nn.Linear(d, features))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        for i, conv in enumerate(self.convs):
            fea = conv(x).unsqueeze_(dim=1)
            if i == 0:
                feas = fea
            else:
                feas = torch.cat([feas, fea], dim=1)
        fea_U = torch.sum(feas, dim=1)
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            # print(i, fea_z.shape)
            vector = fc(fea_z).unsqueeze_(dim=1)
            # print(i, vector.shape)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector],
                                              dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v

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


class SDFF(nn.Module):
    # Soft-gating Dual Feature Fusion.

    def __init__(self, in_channels, out_channels):
        super(SDFF, self).__init__()

        self.structure_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            # SELayer(out_channels),
            ECABlock(channels = in_channels + in_channels),
            # SKConv(features=out_channels,WH=1, M=2, G=1, r=2),
            nn.Sigmoid()
        )
        self.detail_branch = nn.Sequential(
            nn.Conv2d(in_channels=in_channels + in_channels, out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1),
            # SELayer(out_channels),
            ECABlock(channels=in_channels + in_channels),
            # SKConv(features=out_channels, WH=1, M=2, G=1, r=2),
            nn.Sigmoid()
        )

        self.structure_gamma = nn.Parameter(torch.zeros(1))
        self.structure_beta = nn.Parameter(torch.zeros(1))
        self.detail_gamma = nn.Parameter(torch.zeros(1))

        self.detail_beta = nn.Parameter(torch.zeros(1))

# 前向传播函数中输入结构特征图和纹理特征图，即Fcst与Fcte。
    def forward(self, structure_feature, detail_feature):
        sd_cat = torch.cat((structure_feature, detail_feature), dim=1)

        map_detail = self.structure_branch(sd_cat)
        map_structure = self.detail_branch(sd_cat)

        detail_feature_branch = detail_feature + self.detail_beta * (structure_feature * (self.detail_gamma * (map_detail * detail_feature)))
        structure_feature_branch = structure_feature + self.structure_beta*(detail_feature*(self.structure_gamma * (map_structure * detail_feature)))
        mapping_image(detail_feature_branch,256,'detail_feature_branch')
        mapping_image(structure_feature_branch,256,'structure_feature_branch')
        return torch.cat((structure_feature_branch, detail_feature_branch), dim=1)


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

