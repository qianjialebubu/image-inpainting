
import time
import cv2
import numpy
import numpy as np
from PIL import Image
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFileDialog

from deeplearing.fixpic import fix

from PyQt5 import QtCore, QtGui, QtWidgets

# 打开文件图标对应选择图片
def clicked_fileButton(self):
    fileName, filetype = QFileDialog.getOpenFileName(
        self,
        "选取文件",
        "e:/",
        "Image Files (*.bmp *.jpg *.jpeg *.png);;Text Files (*.txt)")
    # python 3.x 将系统字符编码默认为了Unicode，而opencv 读取图片函数的输入参数默认用gbk格式处理
    # srcImage = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    # srcImage = cv2.imread(fileName)
    self.fileName_input = fileName

    srcImage = cv2.imdecode(np.fromfile(fileName, dtype=np.uint8), -1)
    MyMatImage.srcImage = srcImage
    image_height, image_width, image_depth = srcImage.shape  # 获取图像的高，宽以及深度。
    # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
    QImg = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
    QShowImage = QImage(QImg.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                        image_width * image_depth,
                        QImage.Format_RGB888)
    self.label_show.clear()
    QShowImage = QShowImage.scaled(
        self.label_show.width(),
        self.label_show.height())  # 图片适应label大小
    self.label_show.setPixmap(QPixmap.fromImage(QShowImage))
# 点击修复按钮后台进行修复图像
def clicked_importButton(self):

    # 分析出掩膜
    root_imc = self.fileName_input
    # 纯黑图像
    img_c = cv2.imread("./image/0.png")
    # img_c = cv2.imread("E:/deeplearing/pyqtv/other/image1/0.png")
    img_b = cv2.imread(root_imc)
    # img_b = cv2.imread("D:/file/work_space/deep_leaning/pythonProject1/add_image/place2/img_mask/91_1.png")
    img_b = cv2.resize(img_b, (256, 256))
    # xishu = 1
    # print(img_b)
    for i in range(255):
        for j in range(255):
            if (img_b[i][j][0] == 255):
                img_c[i][j] = [255, 255, 255]
    img_c = img_c.astype(np.uint8)
    img_c = np.clip(img_c, 0, 255)
    # cv2.waitKey()
    # cv2.imwrite(root_out, img_c)
    file_path = self.fileName_input
    # directory = os.path.dirname(file_path)  # 获取目录路径
    # # 需要numpy转PIL
    img_c = Image.fromarray(img_c)
    start_time = time.time()
    pt_imageroot= fix(path_d=self.fileName_input,
                       mask_image=img_c)
    end_time = time.time()
    print(start_time)
    print(end_time)
    t = "%.{}f".format(2) % (end_time-start_time)
    self.resultWidget.addItem("执行时间"+str(t)+"秒")
    self.outimage1 = pt_imageroot
    srcImage = cv2.imdecode(np.fromfile(self.outimage1, dtype=np.uint8), -1)
    MyMatImage.srcImage = srcImage
    image_height, image_width, image_depth = srcImage.shape  # 获取图像的高，宽以及深度。
    # opencv读图片是BGR，qt显示要RGB，所以需要转换一下
    QImg = cv2.cvtColor(srcImage, cv2.COLOR_BGR2RGB)
    QShowImage = QImage(QImg.data, image_width, image_height,  # 创建QImage格式的图像，并读入图像信息
                        image_width * image_depth,
                        QImage.Format_RGB888)
    self.label_detect.clear()
    QShowImage = QShowImage.scaled(
        self.label_detect.width(),
        self.label_detect.height())  # 图片适应label大小
    self.label_detect.setPixmap(QPixmap.fromImage(QShowImage))
# 点击保存按钮进行保存修复之后的图像
def clicked_importButton_2(self):
    # 获得包含文件路径+文件名的元组
    dirpath = QFileDialog.getSaveFileName(self.label_detect, '选择保存路径', 'e:\\', "Image Files (*.png *.jpg *.jpeg );;Text Files (*.txt)")
    img_save = cv2.imread(self.outimage1)
    cv2.imwrite(dirpath[0], img_save)
# 对应清空按钮
def clicked_importButton_3(self):
    self.label_detect.clear()
    self.label_show.clear()
    _translate = QtCore.QCoreApplication.translate
    self.label_show.setText(_translate("mainWindow", "破损图像"))
    self.label_detect.setText(_translate("mainWindow", "修复图像"))
    self.resultWidget.clear()