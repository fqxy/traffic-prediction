# -*- coding: utf-8 -*-
# !/usr/bin/env python

"""
处理原始数据集：对数据进行归一化处理并转换为输入张量的形式
"""
from math import floor
import pandas as pd
import numpy as np
from sklearn import preprocessing
import glob

# 读取所有.txt 文件的列
def read_files(station_list):
    """
    :param allframe: 所有文件夹里的数据帧
    :param frame: 某个文件夹里的数据帧
    :return: 所有的数据帧拼接而成的数据帧
    """
    list_ = []
    # 读33个文件，station1 ~ station33
    for i in station_list:
        path = r'./PeMS_rename/PeMS/station' + str(i) + '/'

        # 按顺序读取每个station里的txt
        allfiles = []
        for j in range(1, 29):
            allfiles.append(path + str(j) + '.txt')

        # 读station里的txt，28个文件，从2017.1.1~2017.7.15，共196天
        frame_ = []
        for file_ in allfiles:
            table = pd.read_table(file_, usecols=[0, 1])
            # 去除0值
            for idx, val in enumerate(table['Flow (Veh/5 Minutes)']):
                if val == 0:
                    table['Flow (Veh/5 Minutes)'][idx] = table['Flow (Veh/5 Minutes)'][idx - 1]

            frame_.append(table)

        # frame存储了196天的数据，每天有288个5分钟
        frame = pd.concat(frame_)


        list_.append(frame)

    # allframes存储了33个station的数据, (1862784, 2), 1862784 = 33站x196天x24小时x12个五分钟
    allframes = pd.concat(list_)
    return allframes

# 读取所有.txt 文件的列
def read_files_1(station_list):

    list_ = []
    # 读33个文件，station1 ~ station33
    for i in station_list:
        cur_file = r'./PeMS/station' + str(i) + '.txt'
        table = pd.read_table(cur_file, usecols=[0, 1])
        # 去除0值
        for idx, val in enumerate(table['Flow (Veh/5 Minutes)']):
            if val == 0:
                table['Flow (Veh/5 Minutes)'][idx] = floor(table['Flow (Veh/5 Minutes)'][idx - 1]*0.7+ \
                                                           table['Flow (Veh/5 Minutes)'][idx - 2]*0.3)

        list_.append(table)

    allframes = pd.concat(list_)
    return allframes

# 按时间序列对数据分组并标准化
def group_by_time(station_list, filename, read = False, saveCSV = False):

    if saveCSV:
        vehicles_by_time = np.load(filename + '.npy')
        np.savetxt(filename + '.csv', vehicles_by_time, delimiter=',')

    if read:
        return np.load(filename + '.npy')

    frame = read_files_1(station_list)
    # 将第一列的格式改为日期
    frame['5 Minutes'] = pd.to_datetime(frame['5 Minutes'], format='%m/%d/%Y %H:%M')

    # value的类型是Series，有56448个索引，每个索引有33个数据
    values = frame.groupby('5 Minutes')['Flow (Veh/5 Minutes)'].apply(list)
    vehicles_by_time = []
    for i in range(len(values)):
        # 最后得到的vehicle是列表，里面有56448个列表，每个列表里有33个元素
        vehicles_by_time.append(values[i])

    # vehicles_npy的形状为(56448, 33)
    vehicles_by_time_npy = np.array(vehicles_by_time)

    np.save(filename, vehicles_by_time_npy)
    return vehicles_by_time

def group_by_station(station_list, vehicles_by_time):

    vehicles_by_station = vehicles_by_time.transpose()
    vehicles_by_station_npy = np.array(vehicles_by_station)

    file_name = 'data_npy/by_station'
    for s in station_list:
        file_name = file_name + '_' + str(s)

    np.save(file_name, vehicles_by_station_npy)
    return vehicles_by_station



station_list = [1]
filename = 'data_npy/by_time'
for s in station_list:
    filename = filename + '_' + str(s)

vehicles_by_time = group_by_time(station_list,
                                 filename='data_npy/by_time_1',
                                 read=True,
                                 saveCSV=False)


# vehicles_by_staion = group_by_station(station_list, vehicles_by_time)


# vehicles_by_staion = np.load('data_npy/vehicles_by_station.npy')
# vehicles_by_station = vehicles_by_time.transpose()

scaler = preprocessing.MinMaxScaler()

samples_by_time = scaler.fit_transform(vehicles_by_time)
# np.savetxt(filename + '_scaler' +  '.csv', samples_by_time, delimiter=',')

# samples_by_station = samples_by_time.transpose()

"""
if __name__ == "__main__":
    save = group_by_time()
    print(save) 
"""
