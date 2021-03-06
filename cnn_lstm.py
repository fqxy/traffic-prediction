# -*- coding: utf-8 -*-
# !/usr/bin/env python

from keras.models import model_from_json
from keras.models import Sequential
from keras.layers import Dense, Activation, Conv1D, Dropout, Flatten, LSTM
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import keras.backend as K
import input_data

# Convolution
kernel_size = [2, 3, 4]

# Training
time_steps = 8
batch_size = 16
epochs = 15

# 载入模型
print('loading data...')

pems = input_data.read_data_sets()

# 训练集
flow_train = pems.train.flow
flow_train = flow_train.reshape((flow_train.shape[0], time_steps, 33, 1))
labels_train = pems.train.labels

# 测试集
flow_test = pems.test.flow
flow_test = flow_test.reshape((flow_test.shape[0], time_steps, 33, 1))
labels_test = pems.test.labels

# 验证集
flow_validation = pems.validation.flow
flow_validation = flow_validation.reshape((flow_validation.shape[0], time_steps, 33, 1))
labels_validation = pems.validation.labels

# 构建模型
print('build model...')
model = Sequential()

# 第一个卷积层
# 有None（未知）个样本，每个样本input_dim = (time_steps, 33, 1)
# 现对每步时间都做卷积，用TimeDistributed(CNN_model, input_shape = input_dim)
# 卷积的操作本质上是三维的
model.add(TimeDistributed(Conv1D(filters=40,
                                 kernel_size=kernel_size[1],
                                 strides=1,
                                 padding='valid'),
                          input_shape=[time_steps, 33, 1]))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

# 第二个卷积层
model.add(TimeDistributed(Conv1D(40, kernel_size[1], padding='valid')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

# 第三个卷积层
model.add(TimeDistributed(Conv1D(40, kernel_size[0], padding='valid')))
model.add(TimeDistributed(BatchNormalization()))
model.add(TimeDistributed(Activation('relu')))

# 一维化
model.add(TimeDistributed(Flatten()))
model.add(Dense(33))
model.add(Dropout(0.5))

model.add(LSTM(64, return_sequences=True))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(LSTM(256))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(33))

# 打印出网络结构
model.summary()

# 产生网络拓扑图
plot_model(model, to_file='plotModel/cnn_lstm.png')

exit()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# time_distributed_1 (TimeDist (None, 8, 31, 40)         160
# _________________________________________________________________
# time_distributed_2 (TimeDist (None, 8, 31, 40)         160
# _________________________________________________________________
# time_distributed_3 (TimeDist (None, 8, 31, 40)         0
# _________________________________________________________________
# time_distributed_4 (TimeDist (None, 8, 29, 40)         4840
# _________________________________________________________________
# time_distributed_5 (TimeDist (None, 8, 29, 40)         160
# _________________________________________________________________
# time_distributed_6 (TimeDist (None, 8, 29, 40)         0
# _________________________________________________________________
# time_distributed_7 (TimeDist (None, 8, 28, 40)         3240
# _________________________________________________________________
# time_distributed_8 (TimeDist (None, 8, 28, 40)         160
# _________________________________________________________________
# time_distributed_9 (TimeDist (None, 8, 28, 40)         0
# _________________________________________________________________
# time_distributed_10 (TimeDis (None, 8, 1120)           0
# _________________________________________________________________
# dense_1 (Dense)              (None, 8, 33)             36993
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 8, 33)             0
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 8, 64)             25088
# _________________________________________________________________
# activation_4 (Activation)    (None, 8, 64)             0
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 8, 64)             0
# _________________________________________________________________
# lstm_2 (LSTM)                (None, 256)               328704
# _________________________________________________________________
# activation_5 (Activation)    (None, 256)               0
# _________________________________________________________________
# dropout_3 (Dropout)          (None, 256)               0
# _________________________________________________________________
# dense_2 (Dense)              (None, 33)                8481
# =================================================================
# Total params: 407,986
# Trainable params: 407,746
# Non-trainable params: 240
# _________________________________________________________________


model.compile(loss='mean_squared_error',
              optimizer='rmsprop',
              metrics=['mae', 'cosine'])

filepath = "myModel/cnn_lstm.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

# 训练
print('Train...')
model.fit(flow_train,
          labels_train,
#          validation_split=0.33,
          epochs=epochs, 
          batch_size=batch_size,
          callbacks=callbacks_list,
          validation_data=(flow_validation, labels_validation))

model.save(filepath)


score = model.evaluate(flow_test, labels_test, batch_size=batch_size)
print('Test score:', score)

# model_json = model.to_json()
# with open("Model/cnn_lstm_final.json", "w") as json_file:
#     json_file.write(model_json)
# print("Save model to disk")


# json_file = open('Model/cnn_lstm.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# cnn_lstm_model = model_from_json(loaded_model_json)
#
# cnn_lstm_model.load_weights("Model/cnn_lstm.h5")

