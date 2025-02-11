import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution1D, Dropout, Conv1D, MaxPool1D
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.callbacks import TensorBoard
import os


dic = {i:0 for i in range(510)}
for i in range(10):
    path = os.getcwd()
    # 数据读取
    train = pd.read_csv('3128_3.csv')
    # test = pd.read_csv('D:\\code\\pycode\\laman\\test.csv')

    # encode
    def encode(train):
        label_encoder = LabelEncoder().fit(train.species)
        labels = label_encoder.transform(train.species)
        classes = list(label_encoder.classes_)
        train = train.drop(['species', 'id'], axis=1)
        return train, labels,  classes
    train, labels, classes = encode(train)

    # 标准化数据集
    #scaler = StandardScaler().fit(train.values)
    #scaled_train = scaler.transform(train.values)
    scaled_train = train.values
    #print (scaled_train)

    # 数据集划分验证集test_size比例
    # sss = StratifiedShuffleSplit(test_size=0.001, random_state=19)
    # for train_index, valid_index in sss.split(scaled_train, labels):
    #     X_train, X_valid = scaled_train[train_index], scaled_train[valid_index]
    #     y_train, y_valid = labels[train_index], labels[valid_index]



    X_train = scaled_train
    y_train = labels


    # 通道大小3128
    nb_features = 3128
    nb_class = len(classes)

    # reshape形状 （样本数，通道大小。通道数）
    X_train_r = X_train.reshape(len(X_train), 3128, 1)
    #print (X_train_r)

    # 开始模型构建
    model = Sequential()
    model.add(Conv1D(nb_filter = 32, filter_length=3, input_shape=(3128, 1), padding = 'same'))
    model.add(Activation('relu'))

    # model.add(Conv1D(nb_filter = 256, filter_length = 4, padding = 'same'))
    # model.add(Activation('relu'))

    # model.add(Conv1D(nb_filter = 512, filter_length = 2, padding = 'same'))
    # model.add(Activation('relu'))

    model.add(MaxPool1D(pool_size = 4, strides = 2, padding = "same"))

    # model.add(Conv1D(nb_filter=16, filter_length=2, padding = 'same'))
    # model.add(Activation('relu'))
    # model.add(MaxPool1D(pool_size = 5, strides = 1, padding = "same"))

    model.add(Flatten())
    model.add(Dropout(0.4))
    # model.add(Dense(activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dense(nb_class))
    model.add(Activation('softmax'))
    real_class = y_train
    y_train = np_utils.to_categorical(y_train, nb_class)

    # sgd = SGD(lr=0.01, nesterov=True, decay=1e-6, momentum=0.9)





    

    model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    model.summary()
    nb_epoch = 15
    model.fit(X_train_r, y_train, nb_epoch=nb_epoch,batch_size=32)
    # model.fit(X_train_r,y_train,batch_size=16,epochs=20)
    # score=model.evaluate(X_valid_r, y_valid,batch_size=16)
    pre_class = model.predict_classes(X_train_r)
    # real_class = real_class.tolist()





    compre = [1]*510
    for i in range(len(real_class)):
        if real_class[i] != pre_class[i]:
            compre[i] = 0
    print(compre)


    for i in range(len(compre)):
        if compre[i] == 0:
            dic[i] += 1
print(dic)
with open('result.txt', 'w', encoding = 'UTF-8') as f:
    f.write(str(dic))