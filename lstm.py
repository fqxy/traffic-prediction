# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function

from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import keras.backend as K
import input_data

TIME_STEPS = 8  # 输入的时间步数
INPUT_SIZE = 33  # 每步有多少数据
OUTPUT_SIZE = 33  # 输出的维度
EPOCHS = 20  # 迭代次数
BATCH_SIZE = 64


def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


print('loading data...')

train, validation, test = input_data.read_data_sets()

flow_train = train.flow
labels_train = train.labels

flow_test = test.flow
labels_test = test.labels

flow_validation = validation.flow
labels_validation = validation.labels

print('build model...')

model = Sequential()
model.add(LSTM(input_shape=(TIME_STEPS, INPUT_SIZE),
               output_dim=64,
               return_sequences=True, ))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=256))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(OUTPUT_SIZE))

model.summary()

# model.load_weights("Model/lstm.h5")

# 配置损失函数、优化器、评估函数
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', rmse, 'cosine'])
# model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', 'cosine'])


filepath = "myModel/lstm_epochs_20.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

print('Train...')

model.fit(flow_train,
          labels_train,
          epochs=EPOCHS,
          callbacks=callbacks_list,
          batch_size=BATCH_SIZE,
          validation_data=(flow_validation, labels_validation))

# model.fit(flow_train,labels_train,
#           epochs=10,
#           batch_size= BATCH_SIZE,
#           validation_data=(flow_validation, labels_validation))

model.save(filepath)

score = model.evaluate(flow_test, labels_test, verbose=0)
print('Test socre:', score)
#
# model_json = model.to_json()
# with open("myModel/lstm.json", "w") as json_file:
#     json_file.write(model_json)
# model.save_weights("myModel/lstm.h5")
# print("Save model to disk")
#
# json_file = open('myModel/lstm.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# lstm_model = model_from_json(loaded_model_json)

# lstm_model.load_weights("myModel/lstm.h5")
