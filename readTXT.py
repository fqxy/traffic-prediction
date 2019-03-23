readFile = './1.txt'
writeFile = open('new_1.txt', 'w')   #设置文件对象
for line in open(readFile, 'r'): #设置文件对象并读取每一行文件
    writeFile.write('[Date.UTC('+line[6:10]+','+line[0:2]+','+
                    line[3:5]+','+line[11:13]+','+
                    line[14:16]+'),'+line[17:22]+'],\n')
writeFile.close() #关闭文件
