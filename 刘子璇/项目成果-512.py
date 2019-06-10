from __future__ import print_function
from scipy.optimize import fmin_l_bfgs_b
import sys
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import time
import numpy as np
from PIL import Image
from keras import backend
from keras.applications.vgg16 import VGG16

import cv2
import time
import os



content_weight = 0.025
style_weight = 5.0
height = 512
width = 512

def chuli(b):
    image = Image.open(b)
    image = image.resize((512, 512))
    array = np.asarray(image, dtype='float32')
    array = np.expand_dims(array, axis=0)
    array[:, :, :, 0] -= 103.939
    array[:, :, :, 1] -= 116.779
    array[:, :, :, 2] -= 123.68
    array = array[:, :, :, ::-1]
    return array

def content_loss(content, combination):
    return backend.sum(backend.square(combination - content))


def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, combination):
    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = height * width
    return backend.sum(backend.square(S - C)) / (4. * (channels ** 2) * (size ** 2))

def total_variation_loss(x):
    a = backend.square(x[:, :height - 1, :width - 1, :] - x[:, 1:, :width - 1, :])
    b = backend.square(x[:, :height - 1, :width - 1, :] - x[:, :height - 1, 1:, :])
    return backend.sum(backend.pow(a + b, 1.25))

def style_transfer(pathIn='', pathOut='', model='', width=None, jpg_quality=80):
    '''pathIn=''
    pathOut='/home/liuzixuan/实训/02-用Python快速实现图片的风格迁移/result/'
    model='/home/liuzixuan/实训/02-用Python快速实现图片的风格迁移/models/'
    width=512
    jpg_quality = 99 #0-100，设置输出图片的质量，默认80，越大图片质量越好
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



class myform(QMainWindow):
    #close_signal = pyqtSignal()
    def __init__(self):
        super(myform , self).__init__()
        self.setObjectName("MainWindow")
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        #self.setWindowOpacity(0.5)
        self.initUI()
    def initUI(self):
        self.resize(256, 256)
        btn1 = QPushButton(self)
        btn1.setText("IST")
        btn1.setFixedSize(100, 100)
        btn1.setFont(QFont("华文行楷", 30, QFont.Bold))
        btn1.setStyleSheet("QPushButton{background-image:url(48.jpg);"
                           "border:2px solid;" 
                           "border-radius:50px;"
                           "-moz-border-radius:50px;}")
        btn1.clicked.connect(self.msg)

    def msg(self):
        self.hide()
        self.r = picture()
        self.r.show()

class picture(QMainWindow):
    content_array = np.arange(786432).reshape(1, 512, 512, 3)
    style_array = np.arange(786432).reshape(1, 512, 512, 3)
    combination_image = backend.placeholder((1, height, width, 3))
    x = 0

    def __init__(self):
        super(picture, self).__init__()
        self.initUI()

    def initUI(self):
        self.resize(1800, 800)
        self.setWindowTitle("图片")

        self.label = QLabel(self)
        self.label.setText("内容图片")
        self.label.setFixedSize(512, 512)
        self.label.move(40, 60)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel{background:white;}")

        self.label1 = QLabel(self)
        self.label1.setText("风格图片")
        self.label1.setFixedSize(512, 512)
        self.label1.move(600, 60)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setStyleSheet("QLabel{background:white;}")

        btn1 = QPushButton(self)
        btn1.setText("选择内容")
        btn1.move(200, 15)
        btn1.clicked.connect(self.openimage)

        btn2 = QPushButton(self)
        btn2.setText("选择风格")
        btn2.move(800, 15)
        btn2.clicked.connect(self.openimage)

        btn3 = QPushButton(self)
        btn3.setText("ok")
        btn3.move(1300, 15)
        btn3.clicked.connect(self.vgg)

        btn4 = QPushButton(self)
        btn4.setText("cancel")
        btn4.move(1420, 15)
        btn4.clicked.connect(self.onClick_Button)

        btn5 = QPushButton(self)
        btn5.setText("详细转化")
        btn5.setFont(QFont(""))
        btn5.move(1650, 650)
        btn5.setStyleSheet("QPushButton{background:gray;"
                           "border:2px solid;"
                           "border-radius:10px;"
                           "-moz-border-radius:10px;}")
        btn5.clicked.connect(self.slot_btn_function)

        self.label2 = QLabel(self)
        self.label2.setText("图片")
        self.label2.setFixedSize(512, 512)
        self.label2.move(1150, 60)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setStyleSheet("QLabel{background:white;}")

    def vgg(self):
        input_tensor = backend.concatenate([self.content_array,
                                            self.style_array,
                                            self.combination_image], axis=0)
        model = VGG16(input_tensor=input_tensor, weights='imagenet',
                      include_top=False)
        layers = dict([(layer.name, layer.output) for layer in model.layers])

        total_variation_weight = 1.0
        loss = backend.variable(0.)

        layer_features = layers['block2_conv2']
        content_image_features = layer_features[0, :, :, :]
        combination_features = layer_features[2, :, :, :]

        loss += content_weight * content_loss(content_image_features,
                                              combination_features)

        feature_layers = ['block1_conv2', 'block2_conv2',
                          'block3_conv3', 'block4_conv3',
                          'block5_conv3']

        for layer_name in feature_layers:
            layer_features = layers[layer_name]
            style_features = layer_features[1, :, :, :]
            combination_features = layer_features[2, :, :, :]
            sl = style_loss(style_features, combination_features)
            loss += (style_weight / len(feature_layers)) * sl

        loss += total_variation_weight * total_variation_loss(self.combination_image)

        grads = backend.gradients(loss, self.combination_image)

        outputs = [loss]
        outputs += grads
        f_outputs = backend.function([self.combination_image], outputs)

        def eval_loss_and_grads(x):
            x = x.reshape((1, height, width, 3))
            outs = f_outputs([x])
            loss_value = outs[0]
            grad_values = outs[1].flatten().astype('float64')
            return loss_value, grad_values

        class Evaluator(object):
            def __init__(self):
                self.loss_value = None
                self.grads_values = None

            def loss(self, x):
                assert self.loss_value is None
                loss_value, grad_values = eval_loss_and_grads(x)
                self.loss_value = loss_value
                self.grad_values = grad_values
                return self.loss_value

            def grads(self, x):
                assert self.loss_value is not None
                grad_values = np.copy(self.grad_values)
                self.loss_value = None
                self.grad_values = None
                return grad_values

        evaluator = Evaluator()
        x = np.random.uniform(0, 255, (1, height, width, 3)) - 128.
        iterations = 5
        for i in range(iterations):
            x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(),
                                             fprime=evaluator.grads, maxfun=20)
        x = x.reshape((height, width, 3))
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = np.clip(x, 0, 255).astype('uint8')

        image = Image.fromarray(x)
        image.save('./out.jpg')
        self.label2.setPixmap(QPixmap('./out.jpg'))

    def openimage(self):
        sender = self.sender()
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        a = chuli(imgName)
        if sender.text() == '选择内容':
            self.content_array = a
            self.content_array = backend.variable(self.content_array)
            jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
            self.label.setPixmap(jpg)
        else:
            self.style_array = a
            self.style_array = backend.variable(self.style_array)
            jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label.height())
            self.label1.setPixmap(jpg)

    def onClick_Button(self):
        app = QApplication.instance()
        app.quit()

    def slot_btn_function(self):
        self.hide()
        self.a = picture1()
        self.a.show()


class picture1(QWidget):
    pathIn = ''
    pathOut = '/home/liuzixuan/实训/02-用Python快速实现图片的风格迁移/result'
    rootdir = '/home/liuzixuan/实训/02-用Python快速实现图片的风格迁移/models'

    def __init__(self):
        super(picture1, self).__init__()
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
        # self.label.move(180, 10)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("QLabel{background:white;}")

        self.label1 = QLabel(self)
        self.label1.setText("风格图片")
        self.label1.setFixedSize(512, 512)
        # self.label1.move(900, 0)
        self.label1.setAlignment(Qt.AlignCenter)
        self.label1.setStyleSheet("QLabel{background:white;}")

        self.label2 = QLabel(self)
        self.label2.setText("图片")
        self.label2.setFixedSize(512, 512)
        # self.label2.move(900, 540)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.setStyleSheet("QLabel{background:white;}")

        self.label3 = QLabel('请选择转换风格')
        self.label3.setFixedSize(150, 30)
        self.label3.setAlignment(Qt.AlignCenter)

        self.cb = QComboBox()
        self.cb.move(100, 200)
        self.cb.addItems(['composition_vii', 'la_muse', 'starry_night', 'wave', 'candy', 'feathers',
                          'mosaic', 'the_scream', 'udnie'])

        self.cb.currentIndexChanged.connect(self.selectionChange)

        btn1 = QPushButton(self)
        btn1.setText("选择内容")
        btn1.setFixedSize(100, 30)
        btn1.clicked.connect(self.openimage)

        btn3 = QPushButton(self)
        btn3.setText("ok")
        btn3.setFixedSize(100, 30)
        btn3.clicked.connect(self.transfer)

        btn4 = QPushButton(self)
        btn4.setText("cancel")
        btn4.setFixedSize(100, 30)
        btn4.clicked.connect(self.onClick_Button)

        btn5 = QPushButton(self)
        btn5.setText("详细转化")
        btn5.setFont(QFont(""))
        btn5.move(1650, 650)
        btn5.setStyleSheet("QPushButton{background:gray;"
                           "border:2px solid;"
                           "border-radius:10px;"
                           "-moz-border-radius:10px;}")
        btn5.clicked.connect(self.slot_btn_function)

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
        layout3.addWidget(btn5)

        layout.addWidget(Layout1, 0, 0, 1, 3)
        layout.addWidget(Layout2, 1, 0, 1, 2)
        layout.addWidget(Layout3, 1, 2, 1, 1)

        self.setLayout(layout)

    def selectionChange(self):
        #self.label.setText(self.cb.currentText())
        self.label.adjustSize()
        a = self.cb.currentText()
        self.label1.setPixmap(QPixmap('/home/liuzixuan/实训/02-用Python快速实现图片的风格迁移/models/style_images/' + a + '.jpg')
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
                    # if j < 10:
                    #    pathOutimg = self.pathOut + '/result_img0{:d}.jpg'.format(j)
                    # else:
                    #    pathOutimg = self.pathOut + '/result_img{:d}.jpg'.format(j)

                    style_transfer(self.pathIn, self.pathOut + '/image.jpg', model, self.width)
                    j = j + 1
        self.label2.setPixmap(QPixmap('/home/liuzixuan/实训/02-用Python快速实现图片的风格迁移/result/image.jpg'))

    def openimage(self):
        # sender = self.sender()
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.jpg;;*.png;;All Files(*)")
        a = os.path.abspath(imgName)
        # if sender.text() == '选择内容':
        self.pathIn = a
        # self.pathIn = backend.variable(self.pathIn)
        jpg = QtGui.QPixmap(imgName).scaled(self.label.width(), self.label.height())
        self.label.setPixmap(jpg)
        # else:
        #    self.pathOut = a
        #    #self.style_array = backend.variable(self.style_array)
        #    jpg = QtGui.QPixmap(imgName).scaled(self.label1.width(), self.label.height())
        #    self.label1.setPixmap(jpg)

    def onClick_Button(self):
        app = QApplication.instance()
        app.quit()

    def slot_btn_function(self):
        self.hide()
        self.a = picture()
        self.a.show()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    w = myform()
    w1 = picture()
    w2 = picture1()
    w.show()
    #w.close_signal.connect(w1.show)
    sys.exit(app.exec_())