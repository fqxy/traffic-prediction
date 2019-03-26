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

y1 = vehicles_by_station[16][288:288*2]
y2 = vehicles_by_station[16][288*2:288*3]
y3 = vehicles_by_station[16][288*3:288*4]
y4 = vehicles_by_station[16][288*4:288*5]
y5 = vehicles_by_station[16][288*5:288*6]
y6 = vehicles_by_station[16][288*6:288*7]
y7 = vehicles_by_station[16][288*7:288*8]

matplotlib.rcParams['font.family'] = 'SimHei'

plt.figure(figsize=(7, 7))

plt.subplot(211)
plt.plot(y1)
plt.plot(y2)
plt.plot(y3)
plt.plot(y4)
plt.plot(y5)
plt.legend(('周一', '周二', '周三', '周四', '周五'), loc = 'upper right')
plt.xlabel('时间')
plt.xticks([0, 36, 72, 108, 144, 180, 216, 252, 288],
       ['0:00', '3:00', '6:00', '9:00', '12:00',
        '15:00', '18:00', '21:00', "0:00"])
plt.ylabel('车流量(Veh/5 Minutes)')
plt.grid(True)

plt.subplot(212)
plt.plot(y6, label="周六")
plt.plot(y7, label="周日")
plt.legend(('周六', '周日'), loc = 'upper right')
plt.xlabel('时间')
plt.xticks([0, 36, 72, 108, 144, 180, 216, 252, 288],
       ['0:00', '3:00', '6:00', '9:00', '12:00',
        '15:00', '18:00', '21:00', "0:00"])
plt.ylabel('车流量(Veh/5 Minutes)')
plt.grid(True)

plt.show()