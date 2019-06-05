# -*- coding: utf-8 -*-
from __future__ import print_function
from scipy.optimize import fmin_l_bfgs_b
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
import numpy as np
from PIL import Image
from keras import backend




## 载入所需库
import cv2
import time
import os


def style_transfer(pathIn='', pathOut='', model='', width=None, jpg_quality=80):
    '''
    pathIn: 原始图片的路径
    pathOut: 风格化图片的保存路径
    model: 预训练模型的路径
    width: 设置风格化图片的宽度，默认为None, 即原始图片尺寸
    jpg_quality: 0-100，设置输出图片的质量，默认80，越大图片质量越好
    '''

    ## 读入原始图片，调整图片至所需尺寸，然后获取图片的宽度和高度
    width = 512
    img = cv2.imread(pathIn)
    (h, w) = img.shape[:2]
    if width is not None:
        img = cv2.resize(img, (width, width), interpolation=cv2.INTER_CUBIC)
        (h, w) = img.shape[:2]

    ## 从本地加载预训练模型
    print('加载预训练模型......')
    net = cv2.dnn.readNetFromTorch(model)

    ## 将图片构建成一个blob：设置图片尺寸，将各通道像素值减去平均值（比如ImageNet所有训练样本各通道统计平均值）
    ## 然后执行一次前馈网络计算，并输出计算所需的时间
    blob = cv2.dnn.blobFromImage(img, 1.0, (w, h), (103.939, 116.779, 123.680), swapRB=False, crop=False)
    net.setInput(blob)
    start = time.time()
    output = net.forward()
    end = time.time()
    print("风格迁移花费：{:.2f}秒".format(end - start))

    ## reshape输出结果, 将减去的平均值加回来，并交换各颜色通道
    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 103.939
    output[1] += 116.779
    output[2] += 123.680
    output = output.transpose(1, 2, 0)

    ## 输出风格化后的图片
    cv2.imwrite(pathOut, output, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])



class picture(QWidget):
    pathIn = ''
    pathOut = '/home/wangziwei/styletransfer/result'
    rootdir = '/home/wangziwei/styletransfer/models'

    def __init__(self):
        super(picture, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(600, 800)
        self.setWindowTitle("图片")

        layout = QGridLayout()
        layout1 = QGridLayout()
        layout2 = QGridLayout()
        layout3 = QGridLayout()

        Layout1 = QWidget()
        Layout2 = QWidget()
        Layout3 = QWidget()

        background_color = QColor()
        background_color.setNamedColor('#282821')

        self.label = QLabel(self)
        self.label.setText("内容图片")
        self.label.setFixedSize(512, 512)
        self.label.setAutoFillBackground(True)
        palette = QPalette()
        palette.setColor(QPalette.Window, background_color)
        self.label.setPalette(palette)
        #self.label.move(180, 10)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel{background:white;}")

        self.label1 = QLabel(self)
        self.label1.setText("风格图片")
        self.label1.setFixedSize(512, 512)
        #self.label1.move(900, 0)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setStyleSheet("QLabel{background:white;}")


        self.label2 = QLabel(self)
        self.label2.setText("图片")
        self.label2.setFixedSize(512, 512)
        #self.label2.move(900, 540)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setStyleSheet("QLabel{background:white;}")


        self.label3 = QLabel('请选择转换风格')
        self.label3.setFixedSize(150, 20)

        self.cb = QComboBox()
        self.cb.move(100, 200)
        self.cb.addItems(['composition_vii', 'la_muse', 'starry_night', 'wave', 'candy', 'feathers',
                          'mosaic', 'the_scream', 'udnie'])

        self.cb.currentIndexChanged.connect(self.selectionChange)

        btn1 = QPushButton(self)
        btn1.setText("选择内容")
        btn1.setFixedSize(100, 20)
        btn1.clicked.connect(self.openimage)

        btn3 = QPushButton(self)
        btn3.setText("ok")
        btn3.setFixedSize(100, 20)
        btn3.clicked.connect(self.transfer)

        btn4 = QPushButton(self)
        btn4.setText("cancel")
        btn4.setFixedSize(100, 20)
        btn4.clicked.connect(self.onClick_Button)

        Layout1.setLayout(layout1)
        Layout2.setLayout(layout2)
        Layout3.setLayout(layout3)

        layout1.addWidget(self.label, 0, 0, 1, 1)
        layout1.addWidget(self.label1, 0, 1, 1, 1)
        layout1.addWidget(self.label2, 0, 2, 1, 1)

        layout2.addWidget(btn1, 1, 0, 1, 1)
        layout2.addWidget(self.label3, 0, 1, 1, 1)
        layout2.addWidget(self.cb, 1, 1, 1, 1)

        layout3.addWidget(btn3, 0, 0, 1, 1)
        layout3.addWidget(btn4, 0, 1, 1, 1)

        layout.addWidget(Layout1, 0, 0, 1, 3)
        layout.addWidget(Layout2, 1, 0, 1, 2)
        layout.addWidget(Layout3, 1, 2, 1, 1)



        self.setLayout(layout)






    def selectionChange(self):
        # self.label.setText(self.cb.currentText())
        self.label.adjustSize()
        a = self.cb.currentText()
        self.label1.setPixmap(QPixmap('/home/wangziwei/styletransfer/models/style_images/' + a + '.jpg')
                              .scaled(self.label1.width(), self.label1.height()))
        a = a + '.t7'
        return a

    def transfer(self):
        a = self.selectionChange()
        j = 0
        list = os.listdir(self.rootdir)
        for i in range(0, len(list) - 1):

            list1 = os.path.join(self.rootdir, list[i])
            path = os.listdir(list1)
            for k in range(0, len(path)):
                if path[k] == a:
                    model = './models/' + list[i] + '/' + path[k]
                #if j < 10:
                #    pathOutimg = self.pathOut + '/result_img0{:d}.jpg'.format(j)
                #else:
                #    pathOutimg = self.pathOut + '/result_img{:d}.jpg'.format(j)

                    style_transfer(self.pathIn, self.pathOut + '/image.jpg', model, self.width)
                    j = j + 1
        self.label2.setPixmap(QPixmap('/home/wangziwei/styletransfer/result/image.jpg'))

    def openimage(self):
        #sender = self.sender()
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        a = os.path.abspath(imgName)
        #if sender.text() == '选择内容':
        self.pathIn = a
        #self.pathIn = backend.variable(self.pathIn)
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        #else:
        #    self.pathOut = a
        #    #self.style_array = backend.variable(self.style_array)
        #    jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label.height())
        #    self.label1.setPixmap(jpg)


    def onClick_Button(self):
        app = QApplication.instance()
        app.quit()






if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    my = picture()
    my.show()
    sys.exit(app.exec_())





