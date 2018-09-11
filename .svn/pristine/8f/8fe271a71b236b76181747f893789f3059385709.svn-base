import cv2
import os
from PIL import Image
import numpy as np
from page_recognition.scripts import recognize_digits2


# 先删除img文件夹下文件
def delete():
    path = '../files/img'
    filenames = os.listdir(path)
    for f in filenames:
        c_path = os.path.join(path, f)
        os.remove(c_path)


# 形态学变化的预处理
def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
    # 2. 二值化
    ret, binary = cv2.threshold(sobel, 150, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
    # 3. 膨胀和腐蚀操作的核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 6))

    # 4. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(binary, element2, iterations = 1)

    # 5. 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
    erosion = cv2.erode(dilation, element1, iterations = 1)

    # 6. 再次膨胀，让轮廓明显一些
    dilation2 = cv2.dilate(erosion, element2, iterations = 3)
    return dilation2


# 文字图片查找文字轮廓，计算平均行高
def findTextRegion(img):
    region = []  # 存储轮廓坐标
    list = []  # 存储行高，计算平均行高
    # 1. 查找轮廓
    binary, contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    # 2. 循环每一个轮廓，筛选那些面积小的

    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 800:
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)  # 计算闭合轮廓的周长
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        # print("rect is: ", rect)
        if round(rect[1][1]) < 300:
           list.append(round(rect[1][1]))
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if(height > width * 1.5):
            continue

        region.append(box)
    avg = round(np.mean(list))
    print("行高均值", avg)
    return avg


# 输入扫描的单张图片，进行裁剪成只有页码部分的小图，并进行识别
def crop(bigImage):
    delete()
    # bigImage = 'C:/Users/Administrator/Desktop/test/1018000255.0073.TIF'  # 图像路径
    # 1.读取图像
    img = cv2.imread(bigImage)
    # 2.  转化成灰度图
    GrayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度图
    # 3. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(GrayImage)
    # 4. 查找和筛选文字区域,得到平均行高
    avg_height = findTextRegion(dilation)

    ret, thresh1 = cv2.threshold(GrayImage, 150, 255, cv2.THRESH_BINARY)  # 二值化
    # cv2.imshow('二值化', thresh1)
    # cv2.imwrite('2.png', thresh1)
    # cv2.waitKey(0)
    (x, y) = thresh1.shape
    num = round(x*0.92)  # 可能是页码的区域起始位置，按页码区域占全文的0.08比例算
    print(x, y, '起始：', num)
    k = 0
    a = [0 for z in range(num, x)]   # x个0组成的列表
    for i in range(num, x):
        for j in range(0, y):
            if thresh1[i, j] == 0:  # 黑色
                a[k] = a[k] + 1
                thresh1[i, j] = 255  # to be white
        k += 1
    # print(a)
    # for i in range(num, x):
    #     for j in range(0, a[i-num]):
    #         thresh1[i-num, j] = 0  # 画出相应像素位数个黑色块
    # cv2.imshow('thresh1', thresh1)
    # cv2.waitKey(0)

    result = ''
    # 根据水平投影值选定行分割点 根据垂直投影值选择列分割点
    fg = [[0 for col in range(2)] for row in range(2)]  # 定义一个2行2列的二维列表
    inline = 1
    start = 0
    for i in range(0, a.__len__())[::-1]:
        if inline == 1 and a[i] >= 1:  # 从空白区进入文字区
            start = i  # 记录起始行分割点
            print(i + num)  # 最后一个有图像的像素点
            inline = 0
        else:
            h = start - i
            if (h > 3) and a[i] == 0 and inline == 0:  # 从文字区进入空白区
                inline = 1
                line_height = start - i
                print("行高：", line_height)
                if 15 <= line_height <= avg_height:
                    fg[0][1] = i + num - 2
                    fg[1][1] = start + num + 2
                    # 截取页码区域的行
                    region = (0, i + num - 2, y, start + num + 2)
                    image = Image.open(bigImage)  # 读取图片
                    h_cropImg = image.crop(region)
                    # h_cropImg.show()
                    path = '../files/cropImg.png'
                    h_cropImg.save(path)
                    result = recognize_digits2.recognize(path)
                    # 截出来的部分进行垂直投影
                    # img1 = cv2.imread(path)  # 读取图像
                    # GrayImage1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)  # 灰度图
                    # cv2.imshow('GrayImage1', GrayImage1)
                    # cv2.waitKey(0)
                    # ret1, thresh2 = cv2.threshold(GrayImage1, 150, 255, cv2.THRESH_BINARY)  # 二值化
                    # cv2.imshow('img', thresh2)
                    # cv2.waitKey(0)
                    # (h, w) = thresh2.shape
                    # print('h, w:', h, w)
                    # b = [0 for z in range(0, w)]  # x个0组成的列表
                    # for k in range(0, w):
                    #     for j in range(0, h):
                    #         if thresh2[j, k] == 0:  # 黑色
                    #             b[k] = b[k] + 1
                    #             thresh2[j, k] = 255  # to be white
                    # print('b:', b, 'len(b):', len(b))
                    # for k in range(0, w):
                    #     for j in range(h - b[k], h):
                    #         thresh2[j, k] = 0  # 画出相应像素位数个黑色块
                    # cv2.imshow('img', thresh2)
                    # # img.save('C:/Users/Administrator/Desktop/test/2.png')
                    # cv2.imwrite('thresh2.png', thresh2)
                    # cv2.waitKey(0)
                    break
    if result == '':
        result = '无页码'
    cv2.destroyAllWindows()
    return result


# 输入为整本期刊或者博硕，循环遍历文件夹下每一张图
def files_crop(dir):
    filenames = os.listdir(dir)
    i = 0
    for files in filenames:
        singlefileName = dir + r"/" + files  # 文件路径
        result = crop(singlefileName)
        filename = os.path.splitext(files)[0]  # 文件名
        filetype = os.path.splitext(files)[1]  # 文件扩展名
        new_name = filename + '_' + result
        newdir = os.path.join(dir, new_name + filetype)  # 新的文件路径
        os.rename(singlefileName, newdir)
        i = i+1


if __name__ == '__main__':
    dir = 'C:/Users/Administrator/Desktop/1018042222'
    files_crop(dir)
    # path = 'C:/Users/Administrator/Desktop/1018042170/1018042170.0045.TIF'
    # crop(path)
