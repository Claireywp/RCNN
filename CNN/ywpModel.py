# -*- coding: utf-8 -*-
"""
Created on Thu Jun 01 20:14:58 2017

@author: Wang Tengfei

导入训练之后的模型，对模型进行测试

"""


from __future__ import absolute_import
from __future__ import print_function
from six.moves import range
from keras.models import load_model
import numpy as np
from keras.datasets import cifar10

np.random.seed(1024)  # for reproducibility




#加载数据
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

x = np.empty((2000,3,32,32),dtype="float32")
y = np.empty((2000,),dtype="uint8")



row = 0
for i in range(len(X_test)):
    if y_test[i] == 0 or y_test[i] == 1:
        y[row] = y_test[i]
        x[row,:,:,:] = X_test[i,:,:,:]
        row = row + 1


data = x;

#测试数据归一化
data /= np.max(data)
data -= np.mean(data)

model = load_model('F:\ywpCnn.h5')


#将预测标签与原始标签相减，若结果为零，则认为分类正确
CnnLabel = model.predict_classes( data,batch_size = 1,verbose=1 )-y

#求分类正确数目
LNum = len(CnnLabel)
Sum = 0
for i in range(LNum):
    if CnnLabel[i] == 0:
        Sum += 1

print('\n')
print('正确个数：',Sum)
#正确数目除以测试总数目求得正确率
print('正确率：',(Sum / 2000.0)*100,'%')