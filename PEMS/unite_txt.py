#coding=utf-8
import os

for i in range(1, 34):
    #获取目标文件夹的路径
    filedir = r'./station' + str(i) + '/'

    # 按顺序读取每个station里的txt
    allfiles = []
    for j in range(1, 29):
        allfiles.append(filedir + str(j) + '.txt')

    #打开当前目录下的result.txt文件，如果没有则创建
    f=open('station' + str(i) + '.txt','w')
    f.writelines('5 Minutes	Flow (Veh/5 Minutes)	# Lane Points	% Observed\n')
    #先遍历文件名
    for file in allfiles:
        #遍历单个文件，读取行数
        for line in open(file):
            if '5 Minutes' in line:
                continue
            f.writelines(line)
    #关闭文件
    f.close()