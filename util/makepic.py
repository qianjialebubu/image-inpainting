import os

import cv2
import numpy as np
def addpicmask(mask_root,imageroot,filenamemask,filename):
    print(mask_root)
    print(imageroot)
    print(filenamemask)
    print(filename)

    # 读取原始图片
    original_image = cv2.imread(mask_root)

    # 创建一个黑色背景的目标图片
    target_image = np.zeros((256, 256, 3), np.uint8)

    # 提取白色区域
    lower_white = np.array([200, 200, 200])
    upper_white = np.array([255, 255, 255])
    mask = cv2.inRange(original_image, lower_white, upper_white)

    # 复制白色区域到目标图片
    target_image[mask != 0] = original_image[mask != 0]

    # 保存目标图片
    cv2.imwrite("E:/deeplearing/pyqtv2/util/maskimage/"+filenamemask, target_image)
    # return "E:/deeplearing/pyqtv2/util/maskimage/"+filenamemask

    # img_c原图 root_out输出图片的位置 root_imc：掩膜图
    root_imc = "E:/deeplearing/pyqtv2/util/maskimage/"+filenamemask
    root_out = "E:/deeplearing/pyqtv2/util/output/"+filename
    img_c = cv2.imread(imageroot)
    img_b = cv2.imread(root_imc)
    img_b = cv2.resize(img_b, (256, 256))
    # xishu = 1
    # print(img_b)
    for i in range(255):
        for j in range(255):
            if (img_b[i][j][0] == 255):
                img_c[i][j] = [255, 255, 255]
    img_c = img_c.astype(np.uint8)
    img_c = np.clip(img_c, 0, 255)
    # cv2.imshow('asdf', img_c)

    cv2.waitKey()
    cv2.imwrite(root_out, img_c)
    return root_out