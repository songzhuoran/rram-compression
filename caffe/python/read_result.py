import numpy as np
import re
np.set_printoptions(threshold = 'nan')
f=open('result.txt','r')
data = f.read()
datas = data.split('\n')
i = 0
mat = np.zeros((32,800))
while i <= len(datas):
    search = re.search('Cluster:(\d+)',datas[i])
    if search != None:
        Cluster_num = int(search.group(1))
    else:
        break
    i = i + 1
    search = re.search('total cluster values:(\d+)',datas[i])

    if search != None:
        total_points = int(search.group(1))
    i = i + 1
    j = i
    while i < j + total_points:
        search = re.search('id_layer:(\d+) id_filter:(\d+) id_channel:(\d+)',datas[i])
        if search != None:
            layer_num = int(search.group(1))
            filter_num = int(search.group(2))
            channel_num = int(search.group(3))
        if layer_num == 1:
            mat[filter_num][channel_num] = Cluster_num
        i = i + 1

np.savetxt('layer1.txt',mat,fmt="%3d")


