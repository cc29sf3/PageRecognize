# 将'../files/img'文件夹下的所有.png图片依次识别，最后返回第几页的识别结果
import numpy as np
from PIL import Image
import os
from keras.models import load_model


# 输入单个图片路径，放入训练好的神经网络中进行数字识别，返回是单个数字识别结果
def pred_digit(imgpath):
    # 读取图片转成灰度格式
    img = Image.open(imgpath).convert('L')
    # resize的过程
    if img.size[0] != 28 or img.size[1] != 28:
        img = img.resize((28, 28))
    # 暂存像素值的一维数组
    arr = []

    for i in range(28):
        for j in range(28):
            # mnist 里的颜色是0代表白色（背景），1.0代表黑色
            pixel = 1.0 - float(img.getpixel((j, i))) / 255.0
            # pixel = 255.0 - float(img.getpixel((j, i))) # 如果是0-255的颜色值
            arr.append(pixel)

    arr1 = np.array(arr).reshape((1, 28, 28, 1))

    # 把处理过后的28*28的图像显示出来
    # test_x = arr1.reshape([28, 28])
    # test = Image.new('L', (28, 28), 255)
    # for i in range(28):
    #     for j in range(28):
    #         test.putpixel((j, i), 255 - int(test_x[i][j] * 255.0))
    # test.show()

    model = load_model('../files/my_model.h5')
    preds = model.predict(arr1, verbose=0)
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    # print labels
    name = labels[np.argmax(preds)]
    return name


# 页码分割后单个数字存放在'../files/img'中，遍历该文件夹，依次进行数字识别
def files_recognize():
    dir = '../files/img'
    suffix = 'png'
    filenames = os.listdir(dir)
    filenames.sort(key=lambda x: int(x[:-4]))  # 倒着数第四位'.'为分界线，按照‘.’左边的数字从小到大排序
    i = 0
    page = ''
    for files in filenames:
        singlefileName = dir + r"/" + files
        # print(singlefileName)
        singlefileForm = os.path.splitext(singlefileName)[1][1:]  # 文件后缀名
        if (singlefileForm == suffix):
            digit = pred_digit(singlefileName)
            page += digit
            i = i+1
    print('第'+page+'页')
    return page


if __name__ == '__main__':
    files_recognize()
