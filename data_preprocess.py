# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
处理原始数据集：对数据进行归一化处理并转换为输入张量的形式
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
import glob

FILE_NUMS = 33


# 读取所有.txt 文件的列
def read_files():
    """
    :param allframe: 所有文件夹里的数据帧
    :param frame: 某个文件夹里的数据帧
    :return: 所有的数据帧拼接而成的数据帧
    """
    list_ = []
    # 读33个文件，station1 ~ station33
    for i in range(1, FILE_NUMS+1):
        path = r'./PeMS_rename/PeMS/station' + str(i) + '/'

        # 按顺序读取每个station里的txt
        allfiles = []
        for j in range(1, 29):
            allfiles.append(path + str(j) + '.txt')

        # 读station里的txt，28个文件，从2017.1.1~2017.7.15，共196天
        frame_ = []
        for file_ in allfiles:
            table = pd.read_table(file_, usecols=[0, 1])
            frame_.append(table)

        # frame存储了196天的数据，每天有288个5分钟
        frame = pd.concat(frame_)

        # 去除0值
        for idx, val in enumerate(frame['Flow (Veh/5 Minutes)']):
            if val == 0:
                frame['Flow (Veh/5 Minutes)'][idx] = 5
        list_.append(frame)

    # allframes存储了33个station的数据, (1862784, 2), 1862784 = 33站x196天x24小时x12个五分钟
    allframes = pd.concat(list_)
    return allframes


# 按时间序列对数据分组并标准化
def group_by_time():
    """
    :param key: 按时间分组
    :param zscore: z分数标准化
    :param grouped: 将数据按时间进行分组并标准化后的结果
    :param vehicles: 把数据转换为矩阵形式
    """
    frame = read_files()
    # 将第一列的格式改为日期
    frame['5 Minutes'] = pd.to_datetime(frame['5 Minutes'], format='%m/%d/%Y %H:%M')

    # value的类型是Series，有56448个索引，每个索引有33个数据
    values = frame.groupby('5 Minutes')['Flow (Veh/5 Minutes)'].apply(list)
    vehicles_by_time = []
    for i in range(len(values)):
        # 最后得到的vehicle是列表，里面有56448个列表，每个列表里有33个元素
        vehicles_by_time.append(values[i])
    # vehicles = np.asarray(vehicles)
    # vechicles = vehicles.reshape((196, 288*FILE_NUMS))
    # vechicles = np.array([np.reshape(x, (288, FILE_NUMS)) for x in vechicles])

    # vehicles_npy的形状为(56448, 33)
    vehicles_by_time_npy = np.array(vehicles_by_time)
    np.save('data_npy/vehicles_by_time_no_0.npy', vehicles_by_time_npy)
    return vehicles_by_time

def group_by_station():

    frame = read_files()
    # 将第一列的格式改为日期
    frame['5 Minutes'] = pd.to_datetime(frame['5 Minutes'], format='%m/%d/%Y %H:%M')

    frame2list = frame['Flow (Veh/5 Minutes)'].values.tolist()

    vehicles_by_staion = []
    for i in range(FILE_NUMS):
        vehicles_by_staion.append(frame2list[i*56448 : (i+1)*56448])

    vehicles_by_station_npy = np.array(vehicles_by_staion)
    np.save('data_npy/vehicles_by_station.npy', vehicles_by_station_npy)
    return vehicles_by_staion

# def min_max_scaler(data):
#     a = np.array([])
#     for d in data:


# vehicles_by_time = group_by_time()
# vehicles_by_staion = group_by_station()

vehicles_by_time = np.load('data_npy/vehicles_by_time.npy')
np.savetxt('CSV/vehicles_by_time.csv', vehicles_by_time, delimiter=',')
# vehicles_by_staion = np.load('data_npy/vehicles_by_station.npy')
# vehicles_by_station = vehicles_by_time.transpose()

scaler = preprocessing.MinMaxScaler()

samples_by_time = scaler.fit_transform(vehicles_by_time)
# samples_by_station = samples_by_time.transpose()

"""
if __name__ == "__main__":
    save = group_by_time()
    print(save) 
"""
