import asyncio
import os
from email.header import Header

import oss2
import requests
import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import time
from deeplearing.fixpic2 import fix
import oss
# 邮件
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pymysql

from util.makepic import addpicmask
from util.wximage.incimage import resize_image

app = FastAPI()
# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，也可以指定特定的来源
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头部
)

db = pymysql.connect(host='localhost',
                     user='root',
                     password='abc123',
                     database='fixapp',
                     charset='utf8')

@app.get("/")
async def root():

    return {"message1": "测试输出一张图片的地址，调用阿里云oss",
            "message": "Hello World",
            "url": "https://blog-bu.oss-cn-beijing.aliyuncs.com/fixpic/91_5.png"}
@app.post("/fixPic")
async def fixPic(request: Request):
    data = await request.json()
    url = data.get("data")
    print(url)
    filename = url.split('/')[7].split('?')[0]
    # 得到图片的名字
    print(filename)
    res = requests.get(url)
    pic = res.content
    photo = open(r'deeplearing/image/fix_before/'+filename, 'wb')
    photo.write(pic)
    photo.close()
    fileName_input = 'deeplearing/image/fix_before/'+filename
    root_imc = fileName_input
    img_c = cv2.imread("image/0.png")
    img_b = cv2.imread(root_imc)
    img_b = cv2.resize(img_b, (256, 256))
    for i in range(255):
        for j in range(255):
            if (img_b[i][j][0] == 255):
                img_c[i][j] = [255, 255, 255]
    img_c = img_c.astype(np.uint8)
    img_c = np.clip(img_c, 0, 255)
    img_c = Image.fromarray(img_c)
    pt_imageroot = fix(path_d=fileName_input, mask_image=img_c,fileName =filename )
    print(pt_imageroot)
    str = 'fixpic/' + filename
    print(str)
    # 配置OSS连接信息
    access_key_id = '填写自己的配置信息'
    access_key_secret = '填写自己的配置信息'
    endpoint = '填写自己的配置信息'  # 例如：http://oss-cn-hangzhou.aliyuncs.com
    bucket_name = '填写自己的配置信息'

    # 创建OSS客户端
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    # 上传文件
    # local_file_path = local_file_path
    # oss_file_path = oss_file_path
    # local_file_path = 'E:/deeplearing/pyqtv2/image/91_5.png'
    # oss_file_path = 'fixpic/91_5.png'
    bucket.put_object_from_file(str, pt_imageroot)
    # oss(local_file_path=pt_imageroot,oss_file_path=str)

    return {"message1": "测试输出一张图片的地址，调用阿里云oss",
            "message": "Hello World",
            "url": "https://blog-bu.oss-cn-beijing.aliyuncs.com/fixpic/91_5.png",
            "url_pic": "https://blog-bu.oss-cn-beijing.aliyuncs.com/"+str}

@app.post("/fixPic1")
async def fixPic1(request: Request):
    data = await request.json()
    print(data)
    flag = False
    select_module = data.get('module')
    url = data.get("imageUrl")
    url_mask = data.get('maskoss')
    accessKeyId = '填写自己的配置信息'
    accessKeySecret = '填写自己的配置信息'
    endpoint = '填写自己的配置信息'
    bucketName = '填写自己的配置信息'
    auth = oss2.Auth(accessKeyId, accessKeySecret)
    bucket = oss2.Bucket(auth, endpoint, bucketName)

    filename = url.split('/')[4]
    imageKey = 'blog-bu/'+filename;
    localFilePath = 'E:/deeplearing/pyqtv2/deeplearing/image/fix_before/'+filename
    # Download the image from OSS to local directory
    bucket.get_object_to_file(imageKey, localFilePath)

    fileName_input = 'E:/deeplearing/pyqtv2/deeplearing/image/fix_before/'+filename
    root_imc = fileName_input
    if (url_mask != ""):
        print("使用画笔")
        # 开始使用画笔的逻辑
        filenamemask = url_mask.split('/')[4]
        imagekeymask = 'blog-bu/' + filenamemask
        localFilePathmask = 'E:/deeplearing/pyqtv2/deeplearing/image/maskoss/' + filenamemask
        bucket.get_object_to_file(imagekeymask, localFilePathmask)
        # 在这里就需要把图片变换为256*256大小的

        # 判断是不是正方形
        image = Image.open(fileName_input)
        width, height = image.size
        # mask = mask_image

        if (width != height):
            print("长与宽不一致")
            # print(path_s)
            fileName_input = resize_image(image_path=fileName_input)
            flag = True


        root_imc = addpicmask(mask_root=localFilePathmask, imageroot=fileName_input,filenamemask=filenamemask,filename=filename)
        print(root_imc)

    img_c = cv2.imread("image/0.png")
    img_b = cv2.imread(root_imc)
    img_b = cv2.resize(img_b, (256, 256))
    for i in range(255):
        for j in range(255):
            if (img_b[i][j][0] == 255):
                img_c[i][j] = [255, 255, 255]
                # print("111")
    img_c = img_c.astype(np.uint8)
    img_c = np.clip(img_c, 0, 255)
    img_c = Image.fromarray(img_c)
    print("Start impainting ...")
    pt_imageroot = fix(path_d=fileName_input, mask_image=img_c,fileName =filename, module = select_module)


    print("end impainting ...")
    str = 'fixpic/' + filename
    # 配置OSS连接信息
    access_key_id = '填写自己的配置信息'
    access_key_secret = '填写自己的配置信息'
    endpoint = '填写自己的配置信息'  # 例如：http://oss-cn-hangzhou.aliyuncs.com
    bucket_name = '填写自己的配置信息'
    # 创建OSS客户端
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)

    # 上传文件
    # local_file_path = local_file_path
    # oss_file_path = oss_file_path
    # local_file_path = 'E:/deeplearing/pyqtv2/image/91_5.png'
    # oss_file_path = 'fixpic/91_5.png'
    bucket.put_object_from_file(str, pt_imageroot)
    # oss(local_file_path=pt_imageroot,oss_file_path=str)
    return {"url_pic": "https://blog-bu.oss-cn-beijing.aliyuncs.com/"+str}

@app.post("/sendEmail")
async def sendEmail(request: Request):
    data = await request.json()
    print(data)
    sender_email = data.get("email")  # 发件人邮箱
    # sender_email = '213046812@qq.com'  # 发件人邮箱
    receiver_email = '2077103562@qq.com'  # 收件人邮箱
    message = data.get("message")
    name = data.get("name")
    Occupation = data.get("selectedOccupation")
    subject = '图像修复APP'  # 邮件主题
    print(message)
    print(type(message))
    # content = str(meassage)
        # 邮件正文
    content = '''
            姓名：{}
            职业：{}
            消息：{}
            '''.format(name, Occupation, message)  # 邮件正文
    # 邮件正文
    print(type(content))

    message = MIMEText(content, 'plain', 'utf-8')
    message['From'] = Header(sender_email)
    message['To'] = Header(receiver_email)
    message['Subject'] = Header(subject)

    try:
        smtpObj = smtplib.SMTP_SSL('smtp.qq.com', 465)  # QQ邮箱的SMTP服务器地址和端口
        smtpObj.login(sender_email, '填写自己的邮箱授权码')  # 邮箱授权码，而不是登录密码
        smtpObj.sendmail(sender_email, receiver_email, message.as_string())
        smtpObj.quit()
        print("邮件发送成功")
    except Exception as e:
        print("邮件发送失败:", e)


    return {"message": "ok"}

@app.post("/fixPic2")
async def fixPic2(request: Request):
    data = await request.json()
    url = data.get("imageUrl")
    filename = url.split('/')[4]
    accessKeyId = '填写自己的配置信息'
    accessKeySecret = '填写自己的配置信息'
    endpoint = '填写自己的配置信息'
    bucketName = '填写自己的配置信息'
    auth = oss2.Auth(accessKeyId, accessKeySecret)
    bucket = oss2.Bucket(auth, endpoint, bucketName)
    imageKey = 'blog-bu/'+filename;
    localFilePath = 'E:/deeplearing/pyqtv2/deeplearing/image/fix_before/'+filename
    bucket.get_object_to_file(imageKey, localFilePath)
    fileName_input = 'deeplearing/image/fix_before/'+filename
    root_imc = fileName_input
    img_c = cv2.imread("image/0.png")
    img_b = cv2.imread(root_imc)
    img_b = cv2.resize(img_b, (256, 256))
    for i in range(255):
        for j in range(255):
            if (img_b[i][j][0] == 255):
                img_c[i][j] = [255, 255, 255]
    img_c = img_c.astype(np.uint8)
    img_c = np.clip(img_c, 0, 255)
    img_c = Image.fromarray(img_c)
    pt_imageroot = fix(path_d=fileName_input, mask_image=img_c,fileName =filename )
    print(pt_imageroot)
    str = 'fixpic/' + filename
    print(str)
    # 配置OSS连接信息
    access_key_id = '填写自己的配置信息'
    access_key_secret = '填写自己的配置信息'
    endpoint = '填写自己的配置信息'  # 例如：http://oss-cn-hangzhou.aliyuncs.com
    bucket_name = '填写自己的配置信息'

    # 创建OSS客户端
    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    bucket.put_object_from_file(str, pt_imageroot)
    print("https://blog-bu.oss-cn-beijing.aliyuncs.com/"+str)
    return {"url_pic": "https://blog-bu.oss-cn-beijing.aliyuncs.com/"+str}

def download_from_oss(self, oss_folder_prefix, object_name, local_save_path):
    """拼接本地保存时的文件路径，且保持oss中指定目录以下的路径层级"""
    oss_path_prefix = object_name.split(oss_folder_prefix)[-1]  # oss原始路径,以'/'为路径分隔符
    oss_path_prefix = os.sep.join(oss_path_prefix.strip('/').split('/'))  # 适配win平台
    local_file_path = os.path.join(local_save_path, oss_path_prefix)
    local_file_prefix = local_file_path[:local_file_path.rindex(os.sep)]  # 本地保存文件的前置路径，如果不存在需创建
    if not os.path.exists(local_file_prefix):
        os.makedirs(local_file_prefix)
    self.bucket.get_object_to_file(object_name, local_file_path)

@app.post("/login")
async def login(request:Request):
    data = await request.json()
    cursor = db.cursor()
    username = data.get('username')
    password = data.get('password')
    print(username, password)
    # 使用 execute()  方法执行 SQL 查询
    sql = "SELECT * FROM user WHERE username='"+username+"'";
    cursor.execute(sql)
    # 使用 fetchone() 方法获取单条数据.
    result = cursor.fetchall()
    if(result[0][1]==password):
        return {'code':200}
        # 关闭数据库连接
        db.close()
    else:
        return {'code':201}
        # 关闭数据库连接
        db.close()
@app.post("/insertlog")
async def insertlog(request:Request):
    data = await request.json()
    cursor = db.cursor()
    creattime = data.get('creattime')
    content = data.get('content')

    # 使用 execute()  方法执行 SQL 查询
    sql = "insert into log(creattime,content) values ('"+creattime+"','"+str(content)+"')";
    cursor.execute(sql)
    db.commit()
    cursor.close()
    # 使用 fetchone() 方法获取单条数据.
    # result = cursor.fetchall()
@app.post("/getlog")
async def getlog(request:Request):
    data = await request.json()
    cursor = db.cursor()
    sql = "select * from log ORDER BY creattime DESC"
    cursor.execute(sql)
    # 使用 fetchone() 方法获取单条数据.
    result = cursor.fetchall()
    print(result)
    return {'result':result}
@app.get("/deletelog")
async def deletelog():
    cursor = db.cursor()
    sql = "DELETE FROM log"
    cursor.execute(sql)
    db.commit()
    return {'message': 'All elements deleted from the table'}

# 启动应用程序
if __name__ == "__main__":
        uvicorn.run(app, host="127.0.0.1", port=8000)

