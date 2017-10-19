# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 12:01:03 2017


训练CNN分类模型

"""
#需要导入的各种包
from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D,AveragePooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import range
import random
import numpy as np
from keras.datasets import cifar10

np.random.seed(1024)  # for reproducibility

#加载数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

x = np.empty((10000,3,32,32),dtype="float32")
y = np.empty((10000,),dtype="uint8")


row = 0
for i in range(len(X_train)):
    if y_train[i] == 0 or y_train[i] == 1:
        y[row] = y_train[i]
        x[row,:,:,:] = X_train[i,:,:,:]
        row = row + 1

data = x;

#测试数据归一化
data /= np.max(data)
data -= np.mean(data)


label = y;

#打乱数据
index = [i for i in range(len(data))]
random.shuffle(index)
data = data[index]
label = label[index]
print(data.shape[0], ' samples')

#label为0~1共2个类别，keras要求格式为binary class matrices,转化一下，直接调用keras提供的这个函数
label = np_utils.to_categorical(label, 2)

###############
#开始建立CNN模型
###############

#生成一个model
model = Sequential()

#第一个卷积层，32个卷积核，每个卷积核大小5*5。1表示输入的图片的通道,灰度图为1通道。
#激活函数用relu
#为防止过拟合，还可以在model.add(Activation('relu'))后加上dropout的技巧: model.add(Dropout(0.5))
model.add(Convolution2D(32, 5, 5, border_mode='valid',input_shape=(3,32,32))) 
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))


#第二个卷积层，32个卷积核，每个卷积核大小3*3
#激活函数用tanh
#采用maxpooling，poolsize为(2,2)
model.add(Convolution2D(32, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.2))
#第三个卷积层，64个卷积核，每个卷积核大小3*3
#激活函数用relu
#采用AveragePooling2D，poolsize为(2,2)
model.add(Convolution2D(64, 3, 3, border_mode='valid')) 
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.1))
#全连接层，先将前一层输出的二维特征图flatten为一维的。
#全连接有64个神经元节点,初始化方式为normal
model.add(Flatten())
model.add(Dense(64,init='normal'))
model.add(Activation('relu'))


#Softmax分类，输出是2类别
model.add(Dense(2, init='normal'))
model.add(Activation('softmax'))


#############
#开始训练模型
##############
#使用随机梯度下降法
#使用SGD + momentum
#model.compile里的参数loss就是损失函数(目标函数)
sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])


#调用fit方法，就是一个训练过程. 训练的epoch数设为10，batch_size为100．
#validation_split=0.1，将10%的数据作为验证集。
model.fit(data, label, batch_size=100, nb_epoch=10,shuffle=True,verbose=1,validation_split=0.1)




