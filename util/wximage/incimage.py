from PIL import Image

def resize_image(image_path):
    # 打开原始图像
    image = Image.open(image_path)

    # 计算调整后图像的大小
    width, height = image.size
    max_dimension = max(width, height)
    new_width = int(width * (256 / max_dimension))
    new_height = int(height * (256 / max_dimension))

    # 调整图像大小
    resized_image = image.resize((new_width, new_height), Image.ANTIALIAS)

    # 创建空白图像
    final_image = Image.new("RGB", (256, 256), (0, 0, 0))

    # 计算空白区域的起始位置
    x = (256 - new_width) // 2
    y = (256 - new_height) // 2
    position = (x, y)
    # 将调整后的图像粘贴到空白图像中
    final_image.paste(resized_image, (x, y))
    final_image.save("E:/deeplearing/pyqtv2/util/wximage/image/resized_image.jpg")

    return "E:/deeplearing/pyqtv2/util/wximage/image/resized_image.jpg"

# 输入图像路径
# image_path = "20230718130334.jpg"

# 调整图像大小并补白
# resized_image = resize_image(image_path)

# 保存结果图像
# resized_image.save("resized_image.jpg")