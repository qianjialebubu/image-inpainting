import time
import cv2
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import torchvision
from torch.utils import data
# from torch.utils.tensorboard import SummaryWriter
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms
from ntpath import basename

if __name__ == "__main__":

    img_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    opt = TestOptions().parse()
    model = create_model(opt)

    # Load your trained model
    model.netEN.module.load_state_dict(
        torch.load("E:/deeplearing/pyqtv2/deeplearing/checkpoints/celeba/32_net_EN.pth", map_location={'cuda:1': 'cuda:0'})['net'])
    model.netDE.module.load_state_dict(
        torch.load("E:/deeplearing/pyqtv2/deeplearing/checkpoints/celeba/32_net_DE.pth", map_location={'cuda:1': 'cuda:0'})['net'])
    model.netMEDFE.module.load_state_dict(
        torch.load("E:/deeplearing/pyqtv2/deeplearing/checkpoints/celeba/32_net_MEDFE.pth", map_location={'cuda:1': 'cuda:0'})['net'])
    # model.netEN.module.load_state_dict(torch.load("")['net'])
    # model.netDE.module.load_state_dict(torch.load("")['net'])
    # model.netMEDFE.module.load_state_dict(torch.load("")['net'])

    results_dir = r'E:/deeplearing/pyqtv2/deeplearing/results'
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    mask_paths = glob('{:s}/*'.format(opt.mask_root))
    mask_paths_2 = glob('{:s}/*'.format(opt.mask_root_2))
    de_paths = glob('{:s}/*'.format(opt.de_root))
    st_path = glob('{:s}/*'.format(opt.st_root))
    image_len = len(de_paths)
    for i in tqdm(range(image_len)):
        # only use one mask for all image
        path_m = mask_paths[0]
        path_d = de_paths[i]
        path_s = st_path[i]
        path_m_2 = mask_paths_2[0]
        mask = Image.open(path_m).convert("RGB")
        mask_2 = Image.open(path_m_2).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")

        mask_2 = mask_transform(mask_2)
        mask = mask_transform(mask)
        # # 查看掩膜中缺失图像的大小
        # print(type(mask))
        # print(mask.shape)
        # # 将张量转换为PIL图像
        # image_pil = transforms.ToPILImage()(mask_2)
        #
        # # 将图像转换为灰度图像
        # gray_image = image_pil.convert("L")
        #
        # # 获取图像的像素数据
        # pixels = gray_image.load()
        #
        # # 查找矩形的位置和大小
        # left = gray_image.width
        # right = 0
        # top = gray_image.height
        # bottom = 0
        #
        # for x in range(gray_image.width):
        #     for y in range(gray_image.height):
        #         if pixels[x, y] == 255:  # 白色像素
        #             left = min(left, x)
        #             right = max(right, x)
        #             top = min(top, y)
        #             bottom = max(bottom, y)
        #
        # # 计算矩形的大小
        # width = right - left + 1
        # height = bottom - top + 1
        #
        # # 打印矩形的大小
        # print("矩形大小：{} x {}".format(width, height))





        detail = img_transform(detail)
        structure = img_transform(structure)
        mask = torch.unsqueeze(mask, 0)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure, 0)

        with torch.no_grad():
            model.set_input(detail, structure, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail * (1 - mask)
            fake_image = (fake_out + 1) / 2.0
        output = fake_image.detach().numpy()[0].transpose((1, 2, 0)) * 255
        output = Image.fromarray(output.astype(np.uint8))
        output.save(rf"{opt.results_dir}/{basename(path_d[:-4])}.png")

