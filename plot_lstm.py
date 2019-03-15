# -*- coding: utf-8 -*-

from matplotlib.font_manager import FontProperties
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as md
import matplotlib.ticker as mt
import pandas as pd
import datetime as dt
import data_preprocess
import input_data

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers import LSTM
import keras.backend as K

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred-y_true), axis=-1))

time_steps = 8
BATCH_SIZE = 64
TIME_STEPS = 8
INPUT_SIZE = 33
OUTPUT_SIZE = 33

flow, labels = input_data.create_data_sets()
x_test = flow[45152:]
labels_test = labels[45152:]

model = Sequential()
model.add(LSTM(input_shape=(TIME_STEPS, INPUT_SIZE),
               output_dim=64,
               return_sequences=True,))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(LSTM(output_dim=256))
model.add(Activation('tanh'))
model.add(Dropout(0.5))
model.add(Dense(OUTPUT_SIZE))

model.summary()

# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_1 (LSTM)                (None, 8, 64)             25088
# _________________________________________________________________
# activation_1 (Activation)    (None, 8, 64)             0
# _________________________________________________________________
# dropout_1 (Dropout)          (None, 8, 64)             0
# _________________________________________________________________
# lstm_2 (LSTM)                (None, 256)               328704
# _________________________________________________________________
# activation_2 (Activation)    (None, 256)               0
# _________________________________________________________________
# dropout_2 (Dropout)          (None, 256)               0
# _________________________________________________________________
# dense_1 (Dense)              (None, 33)                8481
# =================================================================
# Total params: 362,273
# Trainable params: 362,273
# Non-trainable params: 0
# _________________________________________________________________

lstm_model = keras.models.load_model("my_lstm.h5")

scaler = data_preprocess.scaler
# 将归一化数据转化为原来的数
y_test = scaler.inverse_transform(labels_test)

# 用模型进行预测
pred = lstm_model.predict(x_test)
# 将预测值转化为原来的数
lstm_pred = scaler.inverse_transform(pred)
print("LSTM MAE:", metrics.mean_absolute_error(y_test, lstm_pred))
print("LSTM RMSE:", np.sqrt(metrics.mean_squared_error(y_test, lstm_pred)))


plt.rcParams['font.sans-serif'] = ['SimHei']  # for Chinese characters
#fig, ax = plt.subplots()
fig = plt.figure(figsize=(8,10))
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)
date_1 = dt.datetime(2017, 6, 7, 0, 0, 0)
date_2 = dt.datetime(2017, 6, 8, 0, 0, 0)
delta = dt.timedelta(minutes=5)
dates = mpl.dates.drange(date_1, date_2, delta)

ax1.xaxis.set_major_locator(md.HourLocator(byhour=range(24), interval=2))
#ax.xaxis.set_major_locator(md.MinuteLocator(byminute=range(60), interval=40))
ax1.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
#plt.xticks(pd.date_range(date_1,date_2,freq='5min'))#时间间隔
ax2.xaxis.set_major_locator(md.HourLocator(byhour=range(24), interval=2))
#ax.xaxis.set_major_locator(md.MinuteLocator(byminute=range(60), interval=40))
ax2.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
ax3.xaxis.set_major_locator(md.HourLocator(byhour=range(24), interval=2))
#ax.xaxis.set_major_locator(md.MinuteLocator(byminute=range(60), interval=40))
ax3.xaxis.set_major_formatter(md.DateFormatter('%H:%M'))
#
# #plt.ylim((0, 900))
# #plt.yticks(np.linspace(0, 900, 10))
#
#
plt.sca(ax2)
plt.title(u'LSTMs')
plt.ylabel(u'车流量')
plt.ylim((0, 900))
plt.yticks(np.linspace(0, 900, 10))
plt.xticks(rotation=30)
l3, = ax2.plot(dates, y_test[56:344, 16], label=u"真实值")
l4, = ax2.plot(dates, lstm_pred[56:344, 16], color='red',label=u"LSTMs")
plt.legend(handles=[l3, l4], loc='upper right')
#
# ax = plt.gca()
# ax.set_xticks(np.linspace(0, 30, 11))
#
# plt.ylim((0, 900))
# plt.yticks(np.linspace(0, 900, 10))
#
plt.xlabel(u'时间')
# # plt.savefig("threemodels.png", dpi=1200)
# #plt.legend(handles=[l1, l2, l3, l4, l5, l6], loc='upper right')
plt.show()

# plt.plot(y_test[56:344, 16])
# plt.plot(lstm_pred[56:344, 16])
# plt.show()
