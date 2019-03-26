# -*- coding: utf-8 -*-
# !/usr/bin/env python

from __future__ import print_function

import keras
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
from keras.layers.wrappers import TimeDistributed
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping
import keras.backend as K
from keras.utils import plot_model

import input_data
import build_model

TIME_STEPS = 8  # 输入的时间步数
INPUT_SIZE = 33  # 每步有多少数据
OUTPUT_SIZE = 33  # 输出的维度
EPOCHS = 10  # 迭代次数
BATCH_SIZE = 64
drop_out = 0.3
filepath = 'myModel/lstm_epochs' + str(EPOCHS) + '_dropout' + str(drop_out) + '_v1' # 保存模型路径
model_filepath = filepath +'.h5'
checkpoint_filepath = filepath + '_checkpoint'
tensorboard_filepath = filepath + '_log'

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
model.add(Dropout(drop_out))
model.add(LSTM(output_dim=256))
model.add(Activation('tanh'))
model.add(Dropout(drop_out))
model.add(Dense(OUTPUT_SIZE))

# 打印出网络结构
model.summary()

# 产生网络拓扑图
# plot_model(model, to_file='plotModel/lstm.png')

# exit()

# model.load_weights("Model/lstm.h5")

# 配置损失函数、优化器、评估函数
model.compile(loss='mse', optimizer='rmsprop', metrics=['mae', rmse, 'cosine'])

checkpoint_callbacks = ModelCheckpoint(filepath = checkpoint_filepath,
                                       monitor='val_loss',
                                       verbose=1,
                                       save_best_only=True,
                                       mode='min')
checkpoint_callbacks_list = [checkpoint_callbacks]
tensorboard_callbacks = TensorBoard(log_dir=tensorboard_filepath,
                                    write_images=1,
                                    histogram_freq=1)
tensorboard_callbacks_list = [tensorboard_callbacks]

# 训练
print('Train...')
history = model.fit(flow_train,labels_train,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    verbose=1,
                    callbacks=tensorboard_callbacks_list,
                    validation_data=(flow_validation, labels_validation))

model.save(model_filepath)

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
