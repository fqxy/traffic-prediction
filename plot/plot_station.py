import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import preprocessing
# 导入 matplotlib 的所有内容（nympy 可以用 np 这个名字来使用）
# from pylab import *

vehicles_by_time = np.load('../data_npy/vehicles_by_time.npy')

scaler = preprocessing.MinMaxScaler()
vehicles_by_time = scaler.fit_transform(vehicles_by_time)

vehicles_by_station = vehicles_by_time.transpose()

y1 = vehicles_by_station[16][288:288*3]
y2 = vehicles_by_station[23][288:288*3]
y3 = vehicles_by_station[18][288:288*3]
y4 = vehicles_by_station[19][288:288*3]
y5 = vehicles_by_station[20][288:288*3]
y6 = vehicles_by_station[21][288:288*3]
y7 = vehicles_by_station[22][288:288*3]

matplotlib.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(10, 4))

plt.plot(y1, linewidth = 1)
plt.plot(y2, linewidth = 1)
plt.plot(y3, linewidth = 1)
plt.plot(y4, linewidth = 1)
plt.plot(y5, linewidth = 1)
plt.legend(('staion_1', 'staion_2', 'staion_3', 'staion_4', 'staion_5'), loc = 'upper right')
plt.xlabel('时间')
plt.xticks([0, 72, 144, 216, 288, 360, 432, 504, 576],
       ['0:00', '6:00', '12:00', '18:00', "0:00",
        '6:00', '12:00', '18:00', "0:00"])
plt.ylabel('车流量(Veh/5 Minutes)')
plt.grid(True)

plt.show()