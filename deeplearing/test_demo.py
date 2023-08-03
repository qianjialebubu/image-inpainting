import random
import time
import pdb
from options.test_options import TestOptions
from data.dataprocess import DataProcess
from models.models import create_model
import torchvision
from torch.utils import data
import os
import torch
from PIL import Image
import numpy as np
from glob import glob
from tqdm import tqdm
import torchvision.transforms as transforms

# 有以下几张图片需要进行输入：原图，手动的掩膜图，输出结果的保存图
if __name__ == "__main__":
    path_m = 'E:/deeplearing/pyqtv2/deeplearing/image/Mask/0.png'
    path_d = 'E:/deeplearing/pyqtv2/deeplearing/image/GT/0.png'
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
    model.netEN.module.load_state_dict(torch.load("E:/deeplearing/med2/MED13/checkpoints/Mutual Encoder-Decoder/56_net_EN.pth",map_location={'cuda:2':'cuda:0'})['net'])
    model.netDE.module.load_state_dict(torch.load("E:/deeplearing/med2/MED13/checkpoints/Mutual Encoder-Decoder/56_net_DE.pth",map_location={'cuda:2':'cuda:0'})['net'])
    model.netMEDFE.module.load_state_dict(torch.load("E:/deeplearing/med2/MED13/checkpoints/Mutual Encoder-Decoder/56_net_MEDFE.pth",map_location={'cuda:2':'cuda:0'})['net'])
    mask_paths = glob('{:s}/*'.format('E:/deeplearing/pyqtv2/deeplearing/image/Mask'))
    de_paths = glob('{:s}/*'.format('E:/deeplearing/pyqtv2/deeplearing/image/GT'))
    mask_len = len(mask_paths)
    image_len = len(de_paths )
    # 待测试图片路径
    # results_dir = r'E:/deeplearing/pyqtv2/deeplearing/image/GT'
    # 掩膜路径
    # mask_path = r'E:/deeplearing/pyqtv2/deeplearing/image/Mask'
    # 测试结果输出路径
    results_dir_test_gt = r'E:/deeplearing/pyqtv2/deeplearing/image/PT'
    for i in tqdm(range(image_len)):
        # mask_pathss = glob('{:s}/*'.format(mask_path))
        # mask_path_len = glob('{:s}/*'.format(mask_path))
        # len_1 = len(mask_path_len)
        # path_m = mask_pathss[random.randint(0,len_1-1)]

        # path_d = de_paths[i]
        # path_s = de_paths[i]
        path_s = path_d
        mask = Image.open(path_m).convert("RGB")
        detail = Image.open(path_d).convert("RGB")
        structure = Image.open(path_s).convert("RGB")
        detail_save = detail
        mask_save = mask
        mask = mask_transform(mask)
        detail = img_transform(detail)
        structure = img_transform(structure)
        mask = torch.unsqueeze(mask, 0)
        detail = torch.unsqueeze(detail, 0)
        structure = torch.unsqueeze(structure,0)

        with torch.no_grad():
            model.set_input(detail, structure, mask)
            model.forward()
            fake_out = model.fake_out
            fake_out = fake_out.detach().cpu() * mask + detail*(1-mask)
            fake_image = (fake_out+1)/2.0
        output = fake_image.detach().numpy()[0].transpose((1, 2, 0))*255
        output = Image.fromarray(output.astype(np.uint8))
        # output.save(rf"{results_dir}/{i}.png")
        detail_save.save(rf"{results_dir_test_gt}/{i}.png")