import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense

import keras
import tensorflow as tf

# LabelEncoder 用来编码输出标签
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
# StratifiedShuffleSplit可以用来把数据集洗牌，并拆分成训练集和验证集
from sklearn.model_selection import StratifiedShuffleSplit
# 我们用的Keras版本是 2.0.1
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Conv1D, MaxPool1D,Bidirectional,BatchNormalization
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint


# 这个数据集是叶子leaf 品种的分类问题，有三个通道，
# 每个通道64个比特位，一个通道代表边界特征，一个通道代表形状特征，最后一个通道代表材质特征，
# 输出是叶子特征标签的预测

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

train = pd.read_csv('D:\\pycode\\laman\\3000.csv')
# test = pd.read_csv('D:\\pycode\\laman\\test.csv')


# def encode(train, test):
#     # 用LabelEncoder为叶子的种类标签编码，labels对象是训练集上的标签列表
#     label_encoder = LabelEncoder().fit(train.species)
#     labels = label_encoder.transform(train.species)
#     classes = list(label_encoder.classes_)
#     # 此处把不必要的训练集和测试集的列删除
#     train = train.drop(['species', 'id'], axis=1)
#     test = test.drop('id', axis=1)
#     return train, labels, test, classes
# train, labels, test, classes = encode(train, test)


def encode(train):
    # 用LabelEncoder为叶子的种类标签编码，labels对象是训练集上的标签列表
    label_encoder = LabelEncoder().fit(train.species)
    labels = label_encoder.transform(train.species)
    classes = list(label_encoder.classes_)
    # 此处把不必要的训练集和测试集的列删除
    train = train.drop(['species', 'id'], axis=1)

    return train, labels,  classes
train, labels, classes = encode(train)
# 这里只是标准化训练集的特征值
scaler = StandardScaler().fit(train.values)
scaled_train = scaler.transform(train.values)

# scaled_train = train.values
print (scaled_train)
print(labels)
# 把数据集拆分成训练集和测试集，测试集占10%
sss = StratifiedShuffleSplit(test_size=0.2, random_state=15 )
for train_index, valid_index in sss.split(scaled_train, labels):
    X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    y_train, y_valid = labels[train_index], labels[valid_index]

print(y_train)
print('8'*20)
print(y_valid)
# 每个输入通道的大小是64位，一共3个通道
nb_features = 3128
nb_class = len(classes)
#  把输入数据集reshape成keras喜欢的格式：（样本数，通道大小，通道数）
# X_train_r = np.zeros((len(X_train), nb_features, 3))
# # 这里的做法是先把所有元素初始化成0之后，再把刚才的数据集中的数据赋值过来
# X_train_r[:, :, 0] = X_train[:, :nb_features]
# X_train_r[:, :, 1] = X_train[:, nb_features:128]
# X_train_r[:, :, 2] = X_train[:, 128:]
# # 验证集也要reshape一下
# X_valid_r = np.zeros((len(X_valid), nb_features, 3))
# X_valid_r[:, :, 0] = X_valid[:, :nb_features]
# X_valid_r[:, :, 1] = X_valid[:, nb_features:128]
# X_valid_r[:, :, 2] = X_valid[:, 128:]



X_train_r = X_train.reshape(len(X_train), 3128, 1)
X_valid_r = X_valid.reshape(len(X_valid), 3128, 1)
y_train = np_utils.to_categorical(y_train, nb_class)
y_valid = np_utils.to_categorical(y_valid, nb_class)

print (X_train_r)

data_dim = 1
timesteps = 3128
num_classes = 5

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Bidirectional(LSTM(units=128,return_sequences=True,activation='tanh'),input_shape=(timesteps, data_dim)))
model.add(Bidirectional(LSTM(128)))
model.add(Dropout(0.4))

# model.add(BatchNormalization())

# model.add(LSTM(256, return_sequences=True,
#                input_shape=(timesteps, data_dim)))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(128, return_sequences=True))  # returns a sequence of vectors of dimension 32
# model.add(LSTM(128))  # return a single vector of dimension 32
model.add(Dense(5, activation='softmax'))

# sgd = SGD(lr=0.001, nesterov=True, decay=p1e-6, momentum=0.9)
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])



model.fit(X_train_r, y_train,
          batch_size=64, epochs=120,
          validation_data=(X_valid_r, y_valid))




history.loss_plot('epoch')



