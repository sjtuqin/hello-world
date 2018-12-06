# 83%
import tensorflow as tf
import numpy as np
import cv2
import os
from keras.datasets import cifar100
from keras.utils import np_utils
from keras.layers import Convolution2D, MaxPooling2D, AveragePooling2D,Convolution3D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import SGD

#----------------------------------------HYPERMETERS----------------------------------------------
I_SHAPE = 128
CHANNEL_NUM = 3
LABEL_SIZE = 101
FRAME = 5
BATCH_SIZE = 200
np.random.seed(0)
tf.set_random_seed(1)


# (x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
# x_train = x_train / 255
# x_test = x_test / 255
# #这就直接转成one-hot了
# y_train = np_utils.to_categorical(y_train, LABEL_SIZE)
# y_test = np_utils.to_categorical(y_test, LABEL_SIZE)

###################
 # 1. 建立CNN模型
###################
model = Sequential()  # 生成一个model
model.add(Convolution2D(
     32, 3, 3, border_mode='valid', input_shape=[I_SHAPE,I_SHAPE,CHANNEL_NUM]))  # C1 卷积层
model.add(Activation('relu'))  # 激活函数：relu, tanh, sigmoid

model.add(Convolution2D(32, 3, 3))  # C2 卷积层
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # S3 池化
model.add(Dropout(0.25))  #


model.add(Convolution2D(64, 3, 3, border_mode='valid')) # C4
model.add(Activation('relu'))


model.add(Convolution2D(64, 3, 3)) # C5
model.add(Activation('relu'))
model.add(AveragePooling2D(pool_size=(2, 2)))  # S6
model.add(Dropout(0.25))


model.add(Flatten())  # bottleneck 瓶颈
model.add(Dense(512))  # F7 全连接层, 512个神经元
model.add(Activation('relu'))  #
model.add(Dropout(0.5))


model.add(Dense(LABEL_SIZE))  # 100 classes
model.add(Activation('softmax'))  # softmax 分类器
model.summary() # 模型小节
print("建模CNN完成 ...")




###################
# 2. 训练CNN模型
###################
sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
#plot(model, to_file='model1.png', show_shapes=True)  # 画模型图


#准备数据
file_name = os.listdir('C:\\myUCF101')  # 长度 = 501788


for epoch in range(500000):
    select_num = np.random.random_integers(0, 501700, size=BATCH_SIZE)
    select_num_test = np.random.random_integers(0, 501700, size=BATCH_SIZE)
    data = np.empty(shape=[BATCH_SIZE, I_SHAPE, I_SHAPE, CHANNEL_NUM])
    label = np.zeros([BATCH_SIZE, LABEL_SIZE])
    test_data = np.empty(shape=[BATCH_SIZE, I_SHAPE, I_SHAPE, CHANNEL_NUM])
    test_label = np.zeros([BATCH_SIZE, LABEL_SIZE])

    for i in range(BATCH_SIZE):
        img = cv2.imread('C:\\myUCF101' + '\\' + file_name[int(select_num[i])])
        # resize the pic
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(I_SHAPE, I_SHAPE))
        img = np.array(img).reshape([I_SHAPE,I_SHAPE,CHANNEL_NUM])
        data[i] = img /255
        # 现在开始搞label
        label_temp = int(file_name[int(select_num[i])].split('_')[0])
        label[i][label_temp] = 1

        img = cv2.imread('C:\\myUCF101' + '\\' + file_name[int(select_num_test[i])])
        # resize the pic
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, dsize=(I_SHAPE, I_SHAPE))
        img = np.array(img).reshape([I_SHAPE, I_SHAPE, CHANNEL_NUM])
        test_data[i] = img / 255
        # 现在开始搞label
        label_temp = int(file_name[int(select_num_test[i])].split('_')[0])
        test_label[i][label_temp] = 1


    cost = model.train_on_batch(data,label)
    # cost = model.fit(data, label)
    # model.fit(data,label)
    # if(epoch % 1 == 0):

    # print(cost)

    score = model.evaluate(test_data, test_label)

    print('epoch:%d  loss+acc %s test:%s' % (epoch, cost,score))


# model.fit(x_train, y_train, batch_size=100, nb_epoch=5000,
#        validation_data=(x_test, y_test))  # 81.34%, 224.08s
# Y_pred = model.predict_proba(x_test, verbose=0)  # Keras预测概率Y_pred
# print(Y_pred[:3, ])  # 取前三张图片的十类预测概率看看
# score = model.evaluate(x_test, y_test, verbose=0) # 评估测试集loss损失和精度acc
# print('测试集 score(val_loss): %.4f' % score[0])  # loss损失
# print('测试集 accuracy: %.4f' % score[1]) # 精度acc
# print("耗时: %.2f seconds ..." % (time.time() - t0))
