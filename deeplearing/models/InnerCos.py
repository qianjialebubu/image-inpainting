import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.transforms as transforms

import deeplearing.util.util as util
class InnerCos(nn.Module):
    # 把256通道的变成3维的
    def __init__(self):
        super(InnerCos, self).__init__()
        self.criterion = nn.L1Loss()
        self.target = None
        self.down_model = nn.Sequential(
            nn.Conv2d(256, 3, kernel_size=1,stride=1, padding=0),
            nn.Tanh()
        )

    def set_target(self, targetde, targetst):
        image_save(targetde, 'targetde_o')
        image_save(targetst, 'targetst_o')
        self.targetst = F.interpolate(targetst, size=(32, 32), mode='bilinear')
        self.targetde = F.interpolate(targetde, size=(32, 32), mode='bilinear')

    def get_target(self):
        return self.target

    def forward(self, in_data):
        loss_co = in_data[1]
        self.ST = self.down_model(loss_co[0])
        self.DE = self.down_model(loss_co[1])
        self.loss = self.criterion(self.ST, self.targetst)+self.criterion(self.DE, self.targetde)
        self.output = in_data[0]

        image_save(self.ST, 'ST_image')
        image_save(self.DE, 'DE_image')
        image_save(self.targetst, 'targetst_image')
        image_save(self.targetde, 'targetde_image')
        return self.output

    def backward(self, retain_graph=True):

        self.loss.backward(retain_graph=retain_graph)
        return self.loss

    def __repr__(self):

        return self.__class__.__name__

def image_save (input,save_name):
    feature_map_cpu = input.cpu()
    # 转换为范围为 [0, 1] 的张量
    feature_map = (feature_map_cpu - feature_map_cpu.min()) / (feature_map_cpu.max() - feature_map_cpu.min())
    # 转换为 PIL 图像对象
    transform = transforms.ToPILImage()
    image = transform(feature_map[0])
    # 保存为图像文件
    image.save(save_name+'.jpg')