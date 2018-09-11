# -*- coding: utf-8 -*-
# 把数字图片预处理成网络能识别的那种图片
# import the necessary packages
from imutils import contours
import imutils
import cv2
from PIL import Image
from page_recognition.scripts import predict


# 思路：粗提取数字区域后将图片转灰度，自适应二值化，提取轮廓，寻找最小矩形边界，
# 判断是否满足预设条件，如宽、高，宽高比。最后将输入的数字分割成单个的小数字供网络识别
def recognize(imgPath):
    image = cv2.imread(imgPath)
    # 通过调整图像大小来预处理图像，将其转换为灰度图像，模糊图像并计算边缘图，
    image = imutils.resize(image, height=45)  # 通过调整图像大小来对图像进行预处理
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将其转换为灰度级，
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)  # 应用高斯模糊与  5×5 内核降低高频噪声。
    edged = cv2.Canny(blurred, 50, 200, 255)  # 计算边缘图
    # cv2.imshow("Edged1", edged)
    # cv2.waitKey(0)

    # 阈值变形图像，然后应用一系列形态学操作来清除阈值图像
    thresh = cv2.threshold(gray, 20, 255,
                           cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]  # 作用：用于获取二元值的灰度图像 像素高于阈值时，给像素赋予新值，否则，赋予另外一种颜色
    # cv2.imshow('thresh', thresh)
    # cv2.waitKey(0)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 2))  # 形态学操作(膨胀腐蚀),要获取结构元素
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)  # 开运算
    # cv2.imshow('thresh', thresh)

    # 在图像中找到轮廓，然后初始化数字轮廓列表
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,  # 只检测外轮廓
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    digitCnts = []

    # for dc in cnts:
    #     (x, y, w, h) = cv2.boundingRect(dc)
    #     cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 描绘边框
    # cv2.imshow('thresh2', thresh)
    # cv2.waitKey(0)
    # 遍历轮廓找到数字区域
    for c in cnts:
        # 计算轮廓的边界框
        (x, y, w, h) = cv2.boundingRect(c)
        print(x, y, w, h)
        # 如果轮廓足够大，它必须是数字
        if w >= 8 and (h >= 20 and h <= 45):
            digitCnts.append(c)
            # cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 255, 0), 1)  # 描绘边框
    # cv2.imshow('thresh2', thresh)
    # 判断一下那个轮廓有无  异常处理
    if digitCnts:
        # 从左到右排列轮廓
        digitCnts = contours.sort_contours(digitCnts,
                                           method="left-to-right")[0]

        i = 1
        # 在digitCnts中循环轮廓，保存成单个图像以便于后续放入模型中识别
        for dc in digitCnts:
            # （x,y）为矩形左上角的坐标，（w,h）是矩形的宽和高
            (x, y, w, h) = cv2.boundingRect(dc)  # 边界矩形
            # print (x, y, w, h)
            # cv2.rectangle(thresh, (x, y), (x + w, y + h), (0, 0, 255), 1)#描绘边框
            # 截取数字并保存
            thresh_img = Image.fromarray(thresh)   # 将二维数组转换为图像
            cropImg = thresh_img.crop((x, y, x + w, y + h))
            path = '../files/img/' + str(i) + '.png'
            cropImg.save(path)
            i = i + 1

        # 在digitCnts中循环轮廓，并绘制图像上的边界框
        for dc in digitCnts:
            (x, y, w, h) = cv2.boundingRect(dc)
            cv2.rectangle(thresh, (x, y), (x + w, y + h), (255, 255, 255), 1)  # 描绘边框
        # cv2.imshow('thresh2', thresh)
        # cv2.waitKey(0)
        '''
        识别数字——通过神经网络
        '''
        page = predict.files_recognize()
    else:
        page = '无页码'
    return page


if __name__ == '__main__':
    path = "C:/Users/Administrator/Desktop/test/46.png"
    recognize(path)
